<script setup lang="ts">
import { computed, ref } from 'vue'

interface ActivityData {
  date: string
  count: number
}

interface Props {
  data: ActivityData[]
  title?: string
  colorScheme?: 'green' | 'blue' | 'purple' | 'orange'
  weeks?: number
  showMonthLabels?: boolean
  showDayLabels?: boolean
  cellSize?: number
  cellGap?: number
}

const props = withDefaults(defineProps<Props>(), {
  title: '',
  colorScheme: 'green',
  weeks: 52,
  showMonthLabels: true,
  showDayLabels: true,
  cellSize: 12,
  cellGap: 3
})

const emit = defineEmits<{
  cellClick: [date: string, count: number]
  cellHover: [date: string, count: number]
}>()

const hoveredCell = ref<{ date: string; count: number; x: number; y: number } | null>(null)

const colorSchemes = {
  green: ['#ebedf0', '#9be9a8', '#40c463', '#30a14e', '#216e39'],
  blue: ['#ebedf0', '#9ecae1', '#4292c6', '#2171b5', '#084594'],
  purple: ['#ebedf0', '#c4b5fd', '#a78bfa', '#8b5cf6', '#6d28d9'],
  orange: ['#ebedf0', '#fed7aa', '#fdba74', '#fb923c', '#ea580c']
}

const colors = computed(() => colorSchemes[props.colorScheme])

const dayLabels = ['', 'Mon', '', 'Wed', '', 'Fri', '']
const monthLabels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

const dataMap = computed(() => {
  const map = new Map<string, number>()
  props.data.forEach(item => {
    map.set(item.date, item.count)
  })
  return map
})

const maxCount = computed(() => {
  if (props.data.length === 0) return 0
  return Math.max(...props.data.map(d => d.count))
})

const getColorLevel = (count: number): number => {
  if (count === 0 || maxCount.value === 0) return 0
  const ratio = count / maxCount.value
  if (ratio <= 0.25) return 1
  if (ratio <= 0.5) return 2
  if (ratio <= 0.75) return 3
  return 4
}

const calendarData = computed(() => {
  const weeks: Array<Array<{ date: string; count: number; dayOfWeek: number }>> = []
  const today = new Date()
  const startDate = new Date(today)
  startDate.setDate(startDate.getDate() - (props.weeks * 7) + 1)

  // Adjust to start from Sunday
  const dayOfWeek = startDate.getDay()
  startDate.setDate(startDate.getDate() - dayOfWeek)

  let currentWeek: Array<{ date: string; count: number; dayOfWeek: number }> = []
  const totalDays = props.weeks * 7 + 7

  for (let i = 0; i < totalDays; i++) {
    const currentDate = new Date(startDate)
    currentDate.setDate(startDate.getDate() + i)

    const dateStr = currentDate.toISOString().split('T')[0]
    const count = dataMap.value.get(dateStr) || 0
    const dow = currentDate.getDay()

    currentWeek.push({ date: dateStr, count, dayOfWeek: dow })

    if (currentWeek.length === 7) {
      weeks.push(currentWeek)
      currentWeek = []
    }
  }

  if (currentWeek.length > 0) {
    weeks.push(currentWeek)
  }

  return weeks.slice(-props.weeks)
})

const monthMarkers = computed(() => {
  const markers: Array<{ label: string; weekIndex: number }> = []
  let lastMonth = -1

  calendarData.value.forEach((week, weekIndex) => {
    const firstDay = week[0]
    if (firstDay) {
      const date = new Date(firstDay.date)
      const month = date.getMonth()
      if (month !== lastMonth) {
        markers.push({ label: monthLabels[month], weekIndex })
        lastMonth = month
      }
    }
  })

  return markers
})

const totalContributions = computed(() => {
  return props.data.reduce((sum, d) => sum + d.count, 0)
})

const handleCellClick = (date: string, count: number) => {
  emit('cellClick', date, count)
}

const handleCellHover = (event: MouseEvent, date: string, count: number) => {
  const rect = (event.target as HTMLElement).getBoundingClientRect()
  hoveredCell.value = {
    date,
    count,
    x: rect.left + rect.width / 2,
    y: rect.top - 10
  }
  emit('cellHover', date, count)
}

const handleCellLeave = () => {
  hoveredCell.value = null
}

const formatDate = (dateStr: string): string => {
  const date = new Date(dateStr)
  return date.toLocaleDateString('en-US', {
    weekday: 'short',
    month: 'short',
    day: 'numeric',
    year: 'numeric'
  })
}
</script>

<template>
  <div class="activity-heatmap">
    <div v-if="title" class="heatmap-header">
      <h3 class="heatmap-title">{{ title }}</h3>
      <span class="heatmap-total">{{ totalContributions }} total activities</span>
    </div>

    <div class="heatmap-container">
      <!-- Day labels -->
      <div v-if="showDayLabels" class="day-labels">
        <div
          v-for="(label, index) in dayLabels"
          :key="index"
          class="day-label"
          :style="{ height: `${cellSize}px`, marginBottom: `${cellGap}px` }"
        >
          {{ label }}
        </div>
      </div>

      <!-- Calendar grid -->
      <div class="calendar-wrapper">
        <!-- Month labels -->
        <div v-if="showMonthLabels" class="month-labels">
          <span
            v-for="marker in monthMarkers"
            :key="marker.weekIndex"
            class="month-label"
            :style="{ left: `${marker.weekIndex * (cellSize + cellGap)}px` }"
          >
            {{ marker.label }}
          </span>
        </div>

        <!-- Cells grid -->
        <div class="calendar-grid">
          <div
            v-for="(week, weekIndex) in calendarData"
            :key="weekIndex"
            class="week-column"
          >
            <div
              v-for="day in week"
              :key="day.date"
              class="day-cell"
              :style="{
                width: `${cellSize}px`,
                height: `${cellSize}px`,
                marginBottom: `${cellGap}px`,
                backgroundColor: colors[getColorLevel(day.count)]
              }"
              @click="handleCellClick(day.date, day.count)"
              @mouseenter="handleCellHover($event, day.date, day.count)"
              @mouseleave="handleCellLeave"
            />
          </div>
        </div>
      </div>
    </div>

    <!-- Legend -->
    <div class="heatmap-legend">
      <span class="legend-label">Less</span>
      <div
        v-for="(color, index) in colors"
        :key="index"
        class="legend-cell"
        :style="{ backgroundColor: color, width: `${cellSize}px`, height: `${cellSize}px` }"
      />
      <span class="legend-label">More</span>
    </div>

    <!-- Tooltip -->
    <Teleport to="body">
      <div
        v-if="hoveredCell"
        class="heatmap-tooltip"
        :style="{
          left: `${hoveredCell.x}px`,
          top: `${hoveredCell.y}px`,
          transform: 'translate(-50%, -100%)'
        }"
      >
        <strong>{{ hoveredCell.count }} activities</strong>
        <span>{{ formatDate(hoveredCell.date) }}</span>
      </div>
    </Teleport>
  </div>
</template>

<style scoped>
.activity-heatmap {
  padding: 16px;
}

.heatmap-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}

.heatmap-title {
  font-size: 16px;
  font-weight: 600;
  color: #1f2937;
  margin: 0;
}

.heatmap-total {
  font-size: 14px;
  color: #6b7280;
}

.heatmap-container {
  display: flex;
  gap: 8px;
  overflow-x: auto;
  padding-bottom: 8px;
}

.day-labels {
  display: flex;
  flex-direction: column;
  padding-top: 24px;
}

.day-label {
  display: flex;
  align-items: center;
  font-size: 10px;
  color: #6b7280;
  width: 28px;
}

.calendar-wrapper {
  position: relative;
}

.month-labels {
  position: relative;
  height: 20px;
  margin-bottom: 4px;
}

.month-label {
  position: absolute;
  font-size: 10px;
  color: #6b7280;
}

.calendar-grid {
  display: flex;
  gap: v-bind('`${cellGap}px`');
}

.week-column {
  display: flex;
  flex-direction: column;
}

.day-cell {
  border-radius: 2px;
  cursor: pointer;
  transition: transform 0.1s ease, opacity 0.1s ease;
}

.day-cell:hover {
  transform: scale(1.2);
  opacity: 0.8;
}

.heatmap-legend {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 4px;
  margin-top: 12px;
}

.legend-label {
  font-size: 11px;
  color: #6b7280;
  margin: 0 4px;
}

.legend-cell {
  border-radius: 2px;
}

.heatmap-tooltip {
  position: fixed;
  background: rgba(0, 0, 0, 0.9);
  color: white;
  padding: 8px 12px;
  border-radius: 6px;
  font-size: 12px;
  z-index: 10000;
  pointer-events: none;
  display: flex;
  flex-direction: column;
  gap: 2px;
  white-space: nowrap;
}

.heatmap-tooltip::after {
  content: '';
  position: absolute;
  top: 100%;
  left: 50%;
  transform: translateX(-50%);
  border: 6px solid transparent;
  border-top-color: rgba(0, 0, 0, 0.9);
}
</style>
