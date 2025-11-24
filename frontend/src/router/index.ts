import { createRouter, createWebHistory, type RouteRecordRaw } from 'vue-router';

const routes: RouteRecordRaw[] = [
  {
    path: '/',
    name: 'Dashboard',
    component: () => import('@/views/DashboardView.vue'),
    meta: {
      title: 'Dashboard',
      requiresAuth: false
    }
  },
  {
    path: '/conversation',
    name: 'Conversation',
    component: () => import('@/views/ConversationView.vue'),
    meta: {
      title: 'Conversation',
      requiresAuth: false
    }
  },
  {
    path: '/analytics',
    name: 'Analytics',
    component: () => import('@/views/AnalyticsView.vue'),
    meta: {
      title: 'Analytics',
      requiresAuth: false
    }
  },
  {
    path: '/goals',
    name: 'Goals',
    component: () => import('@/views/GoalsView.vue'),
    meta: {
      title: 'Goals',
      requiresAuth: false
    }
  },
  {
    path: '/achievements',
    name: 'Achievements',
    component: () => import('@/views/AchievementsView.vue'),
    meta: {
      title: 'Achievements',
      requiresAuth: false
    }
  },
  {
    path: '/settings',
    name: 'Settings',
    component: () => import('@/views/SettingsView.vue'),
    meta: {
      title: 'Settings',
      requiresAuth: false
    }
  },
  {
    path: '/:pathMatch(.*)*',
    name: 'NotFound',
    component: () => import('@/views/NotFoundView.vue'),
    meta: {
      title: 'Page Not Found'
    }
  }
];

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes,
  scrollBehavior(to, _from, savedPosition) {
    if (savedPosition) {
      return savedPosition;
    }
    if (to.hash) {
      return { el: to.hash, behavior: 'smooth' };
    }
    return { top: 0, behavior: 'smooth' };
  }
});

router.beforeEach((to, _from, next) => {
  const title = to.meta.title as string | undefined;
  document.title = title ? `${title} | Voice Learning Agent` : 'Voice Learning Agent';
  next();
});

router.afterEach((to) => {
  if (import.meta.env.DEV) {
    console.log(`[Router] Navigated to: ${to.fullPath}`);
  }
});

export default router;
