use std::sync::{Arc, Mutex, TryLockError};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use super::constants::{CLIENT_ID, ISSUER, REFRESH_MARGIN_MS};
use super::jwt::{TokenResponse, extract_account_id, validate_token_response};
use super::token_store::{CodexTokenStore, StoredAuth};
use crate::auth::AuthStorage;

pub struct CodexAuthManager<S: AuthStorage<StoredAuth>> {
    pub store: CodexTokenStore<S>,
    cached: Arc<Mutex<Option<StoredAuth>>>,
    // Serializes token refreshes (single-flight). The `cached` mutex only
    // guards cache reads/writes; without this lock, N concurrent requests
    // hitting the expiry margin each POST /oauth/token with the SAME
    // (single-use, rotating) refresh token — the first rotates it, the rest
    // get 401 and used to clear_auth(), destroying the winner's fresh tokens.
    // Observed in production as minutes-long all-requests-401 windows during
    // agent fan-outs at token-expiry boundaries.
    refresh_flight: Arc<Mutex<()>>,
    token_endpoint: String,
}

impl<S: AuthStorage<StoredAuth>> CodexAuthManager<S> {
    pub fn new(store: CodexTokenStore<S>) -> Self {
        Self::with_token_endpoint(store, format!("{ISSUER}/oauth/token"))
    }

    fn with_token_endpoint(store: CodexTokenStore<S>, token_endpoint: String) -> Self {
        Self {
            store,
            cached: Arc::new(Mutex::new(None)),
            refresh_flight: Arc::new(Mutex::new(())),
            token_endpoint,
        }
    }

    fn now_ms() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as u64
    }

    pub fn get_auth(&self) -> Result<StoredAuth, anyhow::Error> {
        let cached = {
            let guard = self.cached.lock().map_err(|e| anyhow::anyhow!("{e}"))?;
            guard.clone()
        };
        let stored = match cached {
            Some(ref auth) => auth.clone(),
            None => {
                let loaded = self.store.load_auth()?;
                match loaded {
                    Some(auth) => {
                        let mut guard = self.cached.lock().map_err(|e| anyhow::anyhow!("{e}"))?;
                        *guard = Some(auth.clone());
                        auth
                    }
                    None => {
                        anyhow::bail!("Not authenticated. Run: claude-code-proxy codex auth login");
                    }
                }
            }
        };

        if stored.expires > Self::now_ms() + REFRESH_MARGIN_MS {
            return Ok(stored);
        }

        self.refresh_now(&stored)
    }

    pub fn force_refresh(&self, rejected: &StoredAuth) -> Result<StoredAuth, anyhow::Error> {
        self.refresh_now(rejected)
    }

    fn refresh_now(&self, snapshot: &StoredAuth) -> Result<StoredAuth, anyhow::Error> {
        // Single-flight: hold the flight lock for the whole refresh. Racing
        // callers block here, then discover the winner's tokens on re-check
        // below and return without touching the token endpoint.
        let wait_started = Instant::now();
        let (_flight, waited_for_flight) = match self.refresh_flight.try_lock() {
            Ok(guard) => (guard, false),
            Err(TryLockError::WouldBlock) => {
                tracing::info!(
                    provider = "codex",
                    event = "auth_refresh_wait",
                    "waiting for in-flight token refresh"
                );
                let guard = self
                    .refresh_flight
                    .lock()
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                tracing::info!(
                    provider = "codex",
                    event = "auth_refresh_wait_complete",
                    wait_ms = wait_started.elapsed().as_millis() as u64,
                    "acquired token refresh flight after waiting"
                );
                (guard, true)
            }
            Err(TryLockError::Poisoned(e)) => return Err(anyhow::anyhow!("{e}")),
        };

        // Re-check under the lock: a concurrent flight (or another process
        // sharing the store) may have refreshed while we waited. Reuse only
        // when persisted auth CHANGED from this caller's pre-wait snapshot.
        // Freshness alone is insufficient: force_refresh() is entered because
        // upstream rejected a token that may still have a future expiry.
        let current = match self.store.load_auth()? {
            Some(latest) => {
                if token_identity_changed(&latest, snapshot)
                    && latest.expires > Self::now_ms() + REFRESH_MARGIN_MS
                {
                    let mut guard = self.cached.lock().map_err(|e| anyhow::anyhow!("{e}"))?;
                    *guard = Some(latest.clone());
                    tracing::info!(
                        provider = "codex",
                        event = "auth_refresh_reuse",
                        waited_for_flight,
                        "reusing persisted token rotated by another refresh flight"
                    );
                    return Ok(latest);
                }
                latest
            }
            None => snapshot.clone(),
        };

        if current.refresh.is_empty() {
            anyhow::bail!("No refresh token stored; re-authenticate");
        }

        tracing::info!(
            provider = "codex",
            event = "auth_refresh_start",
            waited_for_flight,
            "starting token refresh request"
        );
        let client = reqwest::blocking::Client::new();
        let form = [
            ("client_id", CLIENT_ID.to_string()),
            ("grant_type", "refresh_token".to_string()),
            ("refresh_token", current.refresh.clone()),
        ];

        let resp = client
            .post(&self.token_endpoint)
            .form(&form)
            .send()
            .map_err(|e| anyhow::anyhow!("refresh network error: {e}"))?;

        let status = resp.status().as_u16();
        if status == 401 || status == 403 {
            // Before destroying auth state: if the store's refresh token has
            // rotated since we read `current`, a concurrent writer (e.g.
            // another process sharing the Keychain entry) beat us — its
            // tokens are good; return them instead of clobbering the store.
            if let Ok(Some(latest)) = self.store.load_auth()
                && token_identity_changed(&latest, &current)
                && latest.expires > Self::now_ms()
            {
                let mut guard = self.cached.lock().map_err(|e| anyhow::anyhow!("{e}"))?;
                *guard = Some(latest.clone());
                tracing::info!(
                    provider = "codex",
                    event = "auth_refresh_reuse_after_unauthorized",
                    "refresh lost a cross-process race; reusing persisted winner token"
                );
                return Ok(latest);
            }
            {
                let mut guard = self.cached.lock().map_err(|e| anyhow::anyhow!("{e}"))?;
                *guard = None;
            }
            let _ = self.store.clear_auth();
            let err_msg = resp
                .text()
                .unwrap_or_else(|_| "Token refresh unauthorized".to_string());
            anyhow::bail!("{err_msg}");
        }

        if !resp.status().is_success() {
            anyhow::bail!("Token refresh failed: {status}");
        }

        let tokens: TokenResponse = resp
            .json()
            .map_err(|e| anyhow::anyhow!("failed to parse token response: {e}"))?;
        validate_token_response(&tokens)?;
        let account_id = extract_account_id(&tokens).or_else(|| current.account_id.clone());
        let expires = Self::now_ms() + (tokens.expires_in.unwrap_or(3600) * 1000);
        let next = StoredAuth {
            access: tokens.access_token,
            refresh: tokens.refresh_token,
            expires,
            account_id,
        };
        self.store.save_auth(next.clone())?;
        {
            let mut guard = self.cached.lock().map_err(|e| anyhow::anyhow!("{e}"))?;
            *guard = Some(next.clone());
        }
        tracing::info!(
            provider = "codex",
            event = "auth_refresh_complete",
            "token refresh completed and persisted"
        );
        Ok(next)
    }

    pub fn persist_initial_tokens(
        &self,
        tokens: &TokenResponse,
    ) -> Result<StoredAuth, anyhow::Error> {
        validate_token_response(tokens)?;
        let account_id = extract_account_id(tokens);
        let expires = Self::now_ms() + (tokens.expires_in.unwrap_or(3600) * 1000);
        let auth = StoredAuth {
            access: tokens.access_token.clone(),
            refresh: tokens.refresh_token.clone(),
            expires,
            account_id,
        };
        self.store.save_auth(auth.clone())?;
        {
            let mut guard = self.cached.lock().map_err(|e| anyhow::anyhow!("{e}"))?;
            *guard = Some(auth.clone());
        }
        Ok(auth)
    }

    pub fn set_cached(&self, auth: StoredAuth) {
        if let Ok(mut guard) = self.cached.lock() {
            *guard = Some(auth);
        }
    }

    pub fn reset_cache(&self) {
        if let Ok(mut guard) = self.cached.lock() {
            *guard = None;
        }
    }
}

fn token_identity_changed(latest: &StoredAuth, snapshot: &StoredAuth) -> bool {
    latest.access != snapshot.access || latest.refresh != snapshot.refresh
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::InMemoryAuthStore;
    use crate::providers::codex::auth::test_http;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc as StdArc, Barrier};

    fn test_store() -> CodexTokenStore<InMemoryAuthStore<StoredAuth>> {
        CodexTokenStore::new(InMemoryAuthStore::new())
    }

    #[test]
    fn get_auth_returns_stored() {
        let store = test_store();
        let auth = StoredAuth {
            access: "test_access".into(),
            refresh: "test_refresh".into(),
            expires: 9999999999999,
            account_id: Some("acct_1".into()),
        };
        store.save_auth(auth.clone()).unwrap();
        let manager = CodexAuthManager::new(store);
        let result = manager.get_auth().unwrap();
        assert_eq!(result.access, "test_access");
        assert_eq!(result.account_id.as_deref(), Some("acct_1"));
    }

    #[test]
    fn get_auth_fails_when_no_auth() {
        let store = test_store();
        let manager = CodexAuthManager::new(store);
        assert!(manager.get_auth().is_err());
        assert!(
            manager
                .get_auth()
                .unwrap_err()
                .to_string()
                .contains("Not authenticated")
        );
    }

    #[test]
    fn refresh_recheck_returns_concurrently_rotated_tokens_without_network() {
        // Cache holds an EXPIRED token; the store already holds a FRESH one
        // (as after a concurrent flight or another process refreshed). The
        // re-check under the flight lock must return the store's tokens and
        // never reach the token endpoint (no HTTP mock exists here — reaching
        // the network would fail the test with a refresh error).
        let store = test_store();
        let fresh = StoredAuth {
            access: "rotated_access".into(),
            refresh: "rotated_refresh".into(),
            expires: 9_999_999_999_999,
            account_id: Some("acct_1".into()),
        };
        store.save_auth(fresh.clone()).unwrap();
        let manager = CodexAuthManager::new(store);
        manager.set_cached(StoredAuth {
            access: "stale_access".into(),
            refresh: "stale_refresh".into(),
            expires: 0, // expired -> get_auth enters refresh_now
            account_id: Some("acct_1".into()),
        });
        let result = manager.get_auth().unwrap();
        assert_eq!(result.access, "rotated_access");
        assert_eq!(result.refresh, "rotated_refresh");
    }

    #[test]
    fn forced_refresh_reuses_token_rotated_since_rejected_snapshot() {
        // This is the cross-process / late-waiter case: persisted auth changed
        // after this caller sent its request. Reuse it without a second
        // rotation (the production endpoint would fail this test if reached).
        let store = test_store();
        let rotated = StoredAuth {
            access: "rotated_access".into(),
            refresh: "rotated_refresh".into(),
            expires: 9_999_999_999_999,
            account_id: Some("acct_1".into()),
        };
        store.save_auth(rotated.clone()).unwrap();
        let manager = CodexAuthManager::new(store);
        let rejected = StoredAuth {
            access: "rejected_access".into(),
            refresh: "rejected_refresh".into(),
            expires: 9_999_999_999_999,
            account_id: Some("acct_1".into()),
        };

        let result = manager.force_refresh(&rejected).unwrap();
        assert_eq!(result, rotated);
    }

    #[test]
    fn token_identity_ignores_expiry_and_account_metadata() {
        let first = StoredAuth {
            access: "access".into(),
            refresh: "refresh".into(),
            expires: 1,
            account_id: None,
        };
        let metadata_only = StoredAuth {
            expires: 9_999_999_999_999,
            account_id: Some("acct_1".into()),
            ..first.clone()
        };
        assert!(!token_identity_changed(&metadata_only, &first));
    }

    #[test]
    fn concurrent_get_auth_reuses_rotated_persisted_tokens() {
        let store = test_store();
        store
            .save_auth(StoredAuth {
                access: "rotated_access".into(),
                refresh: "rotated_refresh".into(),
                expires: 9_999_999_999_999,
                account_id: None,
            })
            .unwrap();
        let manager = StdArc::new(CodexAuthManager::new(store));
        manager.set_cached(StoredAuth {
            access: "stale".into(),
            refresh: "stale".into(),
            expires: 0,
            account_id: None,
        });
        let mut handles = Vec::new();
        for _ in 0..8 {
            let m = StdArc::clone(&manager);
            handles.push(std::thread::spawn(move || m.get_auth().unwrap().access));
        }
        for h in handles {
            assert_eq!(h.join().unwrap(), "rotated_access");
        }
    }

    #[test]
    fn concurrent_forced_refresh_single_flights_one_real_rotation() {
        const CALLERS: usize = 8;
        let refresh_requests = StdArc::new(AtomicUsize::new(0));
        let request_counter = StdArc::clone(&refresh_requests);
        let server = test_http::spawn_mock_server(
            "mock refresh server should become ready",
            move |request| {
                assert!(request.starts_with("POST /oauth/token "));
                assert!(request.contains("refresh_token=initial_refresh"));
                request_counter.fetch_add(1, Ordering::SeqCst);
                // Keep the winner in flight long enough for every other
                // caller to queue behind the shared mutex deterministically.
                std::thread::sleep(std::time::Duration::from_millis(100));
                test_http::json_response(
                    200,
                    r#"{"access_token":"rotated_access","refresh_token":"rotated_refresh","expires_in":3600}"#,
                )
            },
        );

        let store = test_store();
        let rejected = StoredAuth {
            access: "rejected_access".into(),
            refresh: "initial_refresh".into(),
            // Deliberately fresh: a forced refresh must not short-circuit on
            // expiry alone after upstream has rejected this exact token.
            expires: 9_999_999_999_999,
            account_id: Some("acct_1".into()),
        };
        store.save_auth(rejected.clone()).unwrap();
        let manager = StdArc::new(CodexAuthManager::with_token_endpoint(
            store,
            format!("{}/oauth/token", server.url),
        ));
        manager.set_cached(rejected.clone());

        let barrier = StdArc::new(Barrier::new(CALLERS + 1));
        let mut handles = Vec::new();
        for _ in 0..CALLERS {
            let manager = StdArc::clone(&manager);
            let barrier = StdArc::clone(&barrier);
            let rejected = rejected.clone();
            handles.push(std::thread::spawn(move || {
                barrier.wait();
                manager.force_refresh(&rejected).unwrap()
            }));
        }
        barrier.wait();

        for handle in handles {
            let auth = handle.join().unwrap();
            assert_eq!(auth.access, "rotated_access");
            assert_eq!(auth.refresh, "rotated_refresh");
        }
        assert_eq!(refresh_requests.load(Ordering::SeqCst), 1);

        let persisted = manager.store.load_auth().unwrap().unwrap();
        assert_eq!(persisted.access, "rotated_access");
        assert_eq!(persisted.refresh, "rotated_refresh");
    }
}
