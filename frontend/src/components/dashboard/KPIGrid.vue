<script setup lang="ts">
import { computed } from 'vue'
import KPICard from './KPICard.vue'

interface KPIItem {
  id: string
  title: string
  value: string | number
  trend?: number
  trendLabel?: string
  icon?: string
  iconColor?: string
  variant?: 'default' | 'primary' | 'success' | 'warning' | 'danger' | 'info'
  subtitle?: string
  format?: 'number' | 'percentage' | 'currency' | 'duration'
  invertTrend?: boolean
  clickable?: boolean
}

interface Props {
  items: KPIItem[]
  columns?: 2 | 3 | 4 | 5 | 6
  gap?: 'sm' | 'md' | 'lg'
  loading?: boolean
  responsive?: boolean
}

const props = withDefaults(defineProps<Props>(), {
  columns: 4,
  gap: 'md',
  loading: false,
  responsive: true
})

const emit = defineEmits<{
  cardClick: [id: string, item: KPIItem]
}>()

const gridClasses = computed(() => [
  `columns-${props.columns}`,
  `gap-${props.gap}`,
  { responsive: props.responsive }
])

const handleCardClick = (item: KPIItem) => {
  if (item.clickable) {
    emit('cardClick', item.id, item)
  }
}
</script>

<template>
  <div class="kpi-grid" :class="gridClasses">
    <KPICard
      v-for="item in items"
      :key="item.id"
      :title="item.title"
      :value="item.value"
      :trend="item.trend"
      :trend-label="item.trendLabel"
      :icon="item.icon"
      :icon-color="item.iconColor"
      :variant="item.variant"
      :subtitle="item.subtitle"
      :format="item.format"
      :invert-trend="item.invertTrend"
      :clickable="item.clickable"
      :loading="loading"
      @click="handleCardClick(item)"
    />
  </div>
</template>

<style scoped>
.kpi-grid {
  display: grid;
  width: 100%;
}

/* Column variants */
.columns-2 {
  grid-template-columns: repeat(2, 1fr);
}

.columns-3 {
  grid-template-columns: repeat(3, 1fr);
}

.columns-4 {
  grid-template-columns: repeat(4, 1fr);
}

.columns-5 {
  grid-template-columns: repeat(5, 1fr);
}

.columns-6 {
  grid-template-columns: repeat(6, 1fr);
}

/* Gap variants */
.gap-sm {
  gap: 12px;
}

.gap-md {
  gap: 16px;
}

.gap-lg {
  gap: 24px;
}

/* Responsive behavior */
.kpi-grid.responsive.columns-6 {
  @media (max-width: 1400px) {
    grid-template-columns: repeat(4, 1fr);
  }
  @media (max-width: 1100px) {
    grid-template-columns: repeat(3, 1fr);
  }
  @media (max-width: 800px) {
    grid-template-columns: repeat(2, 1fr);
  }
  @media (max-width: 500px) {
    grid-template-columns: 1fr;
  }
}

.kpi-grid.responsive.columns-5 {
  @media (max-width: 1200px) {
    grid-template-columns: repeat(4, 1fr);
  }
  @media (max-width: 1000px) {
    grid-template-columns: repeat(3, 1fr);
  }
  @media (max-width: 750px) {
    grid-template-columns: repeat(2, 1fr);
  }
  @media (max-width: 500px) {
    grid-template-columns: 1fr;
  }
}

.kpi-grid.responsive.columns-4 {
  @media (max-width: 1000px) {
    grid-template-columns: repeat(3, 1fr);
  }
  @media (max-width: 750px) {
    grid-template-columns: repeat(2, 1fr);
  }
  @media (max-width: 500px) {
    grid-template-columns: 1fr;
  }
}

.kpi-grid.responsive.columns-3 {
  @media (max-width: 750px) {
    grid-template-columns: repeat(2, 1fr);
  }
  @media (max-width: 500px) {
    grid-template-columns: 1fr;
  }
}

.kpi-grid.responsive.columns-2 {
  @media (max-width: 500px) {
    grid-template-columns: 1fr;
  }
}
</style>
