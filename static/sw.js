/**
 * Service Worker for PWA - Week 3 Enhanced Version
 *
 * PATTERN: Stale-while-revalidate with IndexedDB backup
 * WHY: Maximum offline capability while keeping data fresh
 *
 * Features:
 * - Offline-first for static resources
 * - Background sync for pending conversations
 * - IndexedDB for conversation data persistence
 * - Smart cache invalidation
 */

const CACHE_VERSION = 'v3.0.0';
const CACHE_NAME = `voice-learn-${CACHE_VERSION}`;
const API_CACHE_NAME = `voice-learn-api-${CACHE_VERSION}`;
const MAX_CACHED_CONVERSATIONS = 100;

// Static resources to cache on install
const STATIC_RESOURCES = [
    '/static/index.html',
    '/static/manifest.json',
    'https://unpkg.com/vue@3/dist/vue.global.js',
    'https://cdn.tailwindcss.com'
];

// API endpoints to cache for offline access
const CACHEABLE_API_PATHS = [
    '/api/stats',
    '/api/offline-manifest',
    '/api/session/'
];

// IndexedDB configuration
const DB_NAME = 'VoiceLearnOffline';
const DB_VERSION = 1;
const STORES = {
    conversations: 'conversations',
    pending: 'pendingSync',
    settings: 'settings'
};


// ============================================
// Service Worker Lifecycle Events
// ============================================

self.addEventListener('install', event => {
    console.log('[SW] Installing version:', CACHE_VERSION);

    event.waitUntil(
        Promise.all([
            // Cache static resources
            caches.open(CACHE_NAME).then(cache => {
                console.log('[SW] Caching static resources');
                return cache.addAll(STATIC_RESOURCES);
            }),
            // Initialize IndexedDB
            initIndexedDB()
        ]).then(() => {
            console.log('[SW] Installation complete');
            return self.skipWaiting();
        })
    );
});

self.addEventListener('activate', event => {
    console.log('[SW] Activating version:', CACHE_VERSION);

    event.waitUntil(
        Promise.all([
            // Clean up old caches
            caches.keys().then(cacheNames => {
                return Promise.all(
                    cacheNames
                        .filter(name => name.startsWith('voice-learn-') && name !== CACHE_NAME && name !== API_CACHE_NAME)
                        .map(name => {
                            console.log('[SW] Deleting old cache:', name);
                            return caches.delete(name);
                        })
                );
            }),
            // Take control of all clients
            self.clients.claim()
        ])
    );
});


// ============================================
// Fetch Event Handler
// ============================================

self.addEventListener('fetch', event => {
    const url = new URL(event.request.url);

    // Skip WebSocket and non-GET requests for caching
    if (url.protocol === 'ws:' || url.protocol === 'wss:') {
        return;
    }

    // Route to appropriate handler
    if (url.pathname.startsWith('/api/')) {
        event.respondWith(handleApiRequest(event.request));
    } else if (url.pathname.startsWith('/static/') || STATIC_RESOURCES.includes(url.href)) {
        event.respondWith(handleStaticRequest(event.request));
    } else {
        event.respondWith(handleGenericRequest(event.request));
    }
});


// ============================================
// Request Handlers
// ============================================

async function handleStaticRequest(request) {
    // Cache-first strategy for static resources
    const cachedResponse = await caches.match(request);
    if (cachedResponse) {
        // Return cached version, but update in background
        updateCacheInBackground(request);
        return cachedResponse;
    }

    // Fetch from network and cache
    try {
        const networkResponse = await fetch(request);
        if (networkResponse.ok) {
            const cache = await caches.open(CACHE_NAME);
            cache.put(request, networkResponse.clone());
        }
        return networkResponse;
    } catch (error) {
        console.error('[SW] Static fetch failed:', error);
        // Return offline fallback page if available
        return caches.match('/static/index.html');
    }
}

async function handleApiRequest(request) {
    const url = new URL(request.url);

    // For POST requests (conversations), try network first
    if (request.method === 'POST') {
        return handleApiPost(request);
    }

    // Stale-while-revalidate for GET requests
    const cachedResponse = await caches.match(request);

    const networkPromise = fetch(request).then(async response => {
        if (response.ok && isCacheableApi(url.pathname)) {
            const cache = await caches.open(API_CACHE_NAME);
            cache.put(request, response.clone());
        }
        return response;
    }).catch(error => {
        console.warn('[SW] API fetch failed:', error);
        return null;
    });

    // Return cached response immediately if available
    if (cachedResponse) {
        // Update cache in background
        networkPromise;
        return cachedResponse;
    }

    // Wait for network if no cache
    const networkResponse = await networkPromise;
    if (networkResponse) {
        return networkResponse;
    }

    // Return offline response
    return new Response(
        JSON.stringify({
            offline: true,
            error: 'You are currently offline. Data may be stale.',
            cached_at: null
        }),
        {
            status: 503,
            headers: { 'Content-Type': 'application/json' }
        }
    );
}

async function handleApiPost(request) {
    try {
        const response = await fetch(request.clone());
        return response;
    } catch (error) {
        console.warn('[SW] POST request failed, queuing for sync:', error);

        // Store for background sync
        const body = await request.clone().json();
        await storePendingRequest(request.url, body);

        // Return offline acknowledgment
        return new Response(
            JSON.stringify({
                offline: true,
                queued: true,
                message: 'Your message has been saved and will sync when online.'
            }),
            {
                status: 202,
                headers: { 'Content-Type': 'application/json' }
            }
        );
    }
}

async function handleGenericRequest(request) {
    try {
        return await fetch(request);
    } catch (error) {
        const cachedResponse = await caches.match(request);
        if (cachedResponse) {
            return cachedResponse;
        }
        // Fallback to index.html for navigation requests
        if (request.mode === 'navigate') {
            return caches.match('/static/index.html');
        }
        throw error;
    }
}


// ============================================
// Background Sync
// ============================================

self.addEventListener('sync', event => {
    console.log('[SW] Sync event:', event.tag);

    if (event.tag === 'sync-conversations') {
        event.waitUntil(syncPendingConversations());
    }
});

async function syncPendingConversations() {
    const db = await openIndexedDB();
    const tx = db.transaction(STORES.pending, 'readwrite');
    const store = tx.objectStore(STORES.pending);

    const pendingRequests = await getAllFromStore(store);

    for (const item of pendingRequests) {
        try {
            const response = await fetch(item.url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(item.body)
            });

            if (response.ok) {
                // Remove from pending queue
                store.delete(item.id);
                console.log('[SW] Synced pending request:', item.id);
            }
        } catch (error) {
            console.error('[SW] Failed to sync:', error);
        }
    }

    await tx.complete;
    db.close();
}


// ============================================
// IndexedDB Helpers
// ============================================

function initIndexedDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, DB_VERSION);

        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve(request.result);

        request.onupgradeneeded = (event) => {
            const db = event.target.result;

            // Conversations store
            if (!db.objectStoreNames.contains(STORES.conversations)) {
                const convStore = db.createObjectStore(STORES.conversations, { keyPath: 'id', autoIncrement: true });
                convStore.createIndex('session_id', 'session_id', { unique: false });
                convStore.createIndex('timestamp', 'timestamp', { unique: false });
            }

            // Pending sync store
            if (!db.objectStoreNames.contains(STORES.pending)) {
                db.createObjectStore(STORES.pending, { keyPath: 'id', autoIncrement: true });
            }

            // Settings store
            if (!db.objectStoreNames.contains(STORES.settings)) {
                db.createObjectStore(STORES.settings, { keyPath: 'key' });
            }
        };
    });
}

function openIndexedDB() {
    return new Promise((resolve, reject) => {
        const request = indexedDB.open(DB_NAME, DB_VERSION);
        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve(request.result);
    });
}

async function storePendingRequest(url, body) {
    const db = await openIndexedDB();
    const tx = db.transaction(STORES.pending, 'readwrite');
    const store = tx.objectStore(STORES.pending);

    await store.add({
        url,
        body,
        timestamp: Date.now()
    });

    await tx.complete;
    db.close();

    // Register for background sync
    if ('sync' in self.registration) {
        await self.registration.sync.register('sync-conversations');
    }
}

function getAllFromStore(store) {
    return new Promise((resolve, reject) => {
        const request = store.getAll();
        request.onerror = () => reject(request.error);
        request.onsuccess = () => resolve(request.result);
    });
}


// ============================================
// Utility Functions
// ============================================

function isCacheableApi(pathname) {
    return CACHEABLE_API_PATHS.some(path => pathname.startsWith(path));
}

function updateCacheInBackground(request) {
    fetch(request).then(response => {
        if (response.ok) {
            caches.open(CACHE_NAME).then(cache => {
                cache.put(request, response);
            });
        }
    }).catch(() => {
        // Ignore network errors during background update
    });
}


// ============================================
// Message Handler for Client Communication
// ============================================

self.addEventListener('message', event => {
    const { type, payload } = event.data || {};

    switch (type) {
        case 'SKIP_WAITING':
            self.skipWaiting();
            break;

        case 'GET_VERSION':
            event.ports[0].postMessage({ version: CACHE_VERSION });
            break;

        case 'CLEAR_CACHE':
            caches.delete(CACHE_NAME).then(() => {
                caches.delete(API_CACHE_NAME);
            }).then(() => {
                event.ports[0].postMessage({ success: true });
            });
            break;

        case 'STORE_CONVERSATION':
            storeConversationOffline(payload).then(() => {
                event.ports[0].postMessage({ success: true });
            }).catch(error => {
                event.ports[0].postMessage({ success: false, error: error.message });
            });
            break;

        case 'GET_OFFLINE_CONVERSATIONS':
            getOfflineConversations(payload?.session_id).then(conversations => {
                event.ports[0].postMessage({ conversations });
            });
            break;

        default:
            console.log('[SW] Unknown message type:', type);
    }
});

async function storeConversationOffline(conversation) {
    const db = await openIndexedDB();
    const tx = db.transaction(STORES.conversations, 'readwrite');
    const store = tx.objectStore(STORES.conversations);

    // Add timestamp if not present
    conversation.timestamp = conversation.timestamp || Date.now();

    await store.add(conversation);

    // Cleanup old conversations if over limit
    const count = await new Promise((resolve, reject) => {
        const countRequest = store.count();
        countRequest.onsuccess = () => resolve(countRequest.result);
        countRequest.onerror = () => reject(countRequest.error);
    });

    if (count > MAX_CACHED_CONVERSATIONS) {
        const oldestRequest = store.index('timestamp').openCursor();
        oldestRequest.onsuccess = (event) => {
            const cursor = event.target.result;
            if (cursor && count > MAX_CACHED_CONVERSATIONS) {
                cursor.delete();
                cursor.continue();
            }
        };
    }

    await tx.complete;
    db.close();
}

async function getOfflineConversations(sessionId) {
    const db = await openIndexedDB();
    const tx = db.transaction(STORES.conversations, 'readonly');
    const store = tx.objectStore(STORES.conversations);

    let conversations;
    if (sessionId) {
        const index = store.index('session_id');
        conversations = await new Promise((resolve, reject) => {
            const request = index.getAll(sessionId);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => reject(request.error);
        });
    } else {
        conversations = await getAllFromStore(store);
    }

    db.close();
    return conversations;
}


console.log('[SW] Service Worker loaded:', CACHE_VERSION);
