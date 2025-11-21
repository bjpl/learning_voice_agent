import { createApp } from 'vue';
import { createPinia } from 'pinia';
import App from './App.vue';
import router from './router';
import './assets/main.css';

const app = createApp(App);

const pinia = createPinia();

app.use(pinia);
app.use(router);

app.config.errorHandler = (err, instance, info) => {
  console.error('Global error:', err);
  console.error('Component:', instance);
  console.error('Error info:', info);
};

app.config.warnHandler = (msg, instance, trace) => {
  console.warn('Vue warning:', msg);
  if (import.meta.env.DEV) {
    console.warn('Trace:', trace);
  }
};

router.isReady().then(() => {
  app.mount('#app');
});

export { app, pinia, router };
