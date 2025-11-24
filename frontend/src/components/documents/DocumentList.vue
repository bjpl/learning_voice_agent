<script setup lang="ts">
/**
 * DocumentList Component
 * List uploaded documents with actions (view, download, delete)
 */

import { ref, computed } from 'vue';

// ============================================================================
// Type Definitions
// ============================================================================

export interface Document {
  id: string;
  name: string;
  size: number;
  type: string;
  extension: string;
  url: string;
  thumbnailUrl?: string;
  pageCount?: number;
  createdAt: string | Date;
  updatedAt?: string | Date;
  status?: 'processing' | 'ready' | 'error';
  metadata?: Record<string, unknown>;
}

export interface DocumentListProps {
  documents: Document[];
  loading?: boolean;
  selectable?: boolean;
  sortable?: boolean;
  searchable?: boolean;
  emptyMessage?: string;
  columns?: ('name' | 'type' | 'size' | 'date' | 'status' | 'actions')[];
}

export type SortField = 'name' | 'size' | 'createdAt' | 'type';
export type SortOrder = 'asc' | 'desc';

// ============================================================================
// Props & Emits
// ============================================================================

const props = withDefaults(defineProps<DocumentListProps>(), {
  loading: false,
  selectable: false,
  sortable: true,
  searchable: true,
  emptyMessage: 'No documents found',
  columns: () => ['name', 'type', 'size', 'date', 'actions']
});

const emit = defineEmits<{
  (e: 'view', document: Document): void;
  (e: 'download', document: Document): void;
  (e: 'delete', document: Document): void;
  (e: 'select', documents: Document[]): void;
}>();

// ============================================================================
// Refs
// ============================================================================

const searchQuery = ref('');
const sortField = ref<SortField>('createdAt');
const sortOrder = ref<SortOrder>('desc');
const selectedIds = ref<Set<string>>(new Set());

// ============================================================================
// Document Type Mappings
// ============================================================================

const documentColors: Record<string, string> = {
  pdf: '#ef4444',
  doc: '#3b82f6',
  docx: '#3b82f6',
  xls: '#10b981',
  xlsx: '#10b981',
  ppt: '#f59e0b',
  pptx: '#f59e0b',
  txt: '#6b7280',
  csv: '#10b981'
};

// ============================================================================
// Computed
// ============================================================================

const filteredDocuments = computed(() => {
  let result = [...props.documents];

  // Search filter
  if (searchQuery.value) {
    const query = searchQuery.value.toLowerCase();
    result = result.filter(doc =>
      doc.name.toLowerCase().includes(query) ||
      doc.extension.toLowerCase().includes(query)
    );
  }

  // Sort
  if (props.sortable) {
    result.sort((a, b) => {
      let comparison = 0;

      switch (sortField.value) {
        case 'name':
          comparison = a.name.localeCompare(b.name);
          break;
        case 'size':
          comparison = a.size - b.size;
          break;
        case 'type':
          comparison = a.extension.localeCompare(b.extension);
          break;
        case 'createdAt':
          comparison = new Date(a.createdAt).getTime() - new Date(b.createdAt).getTime();
          break;
      }

      return sortOrder.value === 'asc' ? comparison : -comparison;
    });
  }

  return result;
});

const selectedDocuments = computed(() => {
  return props.documents.filter(doc => selectedIds.value.has(doc.id));
});

const isAllSelected = computed(() => {
  return filteredDocuments.value.length > 0 &&
    filteredDocuments.value.every(doc => selectedIds.value.has(doc.id));
});

const isSomeSelected = computed(() => {
  return selectedIds.value.size > 0 && !isAllSelected.value;
});

// ============================================================================
// Methods
// ============================================================================

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(date: string | Date): string {
  const d = new Date(date);
  return d.toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric'
  });
}

function getDocumentColor(extension: string): string {
  return documentColors[extension.toLowerCase()] || '#6b7280';
}

function handleSort(field: SortField): void {
  if (!props.sortable) return;

  if (sortField.value === field) {
    sortOrder.value = sortOrder.value === 'asc' ? 'desc' : 'asc';
  } else {
    sortField.value = field;
    sortOrder.value = 'asc';
  }
}

function toggleSelect(document: Document): void {
  if (!props.selectable) return;

  if (selectedIds.value.has(document.id)) {
    selectedIds.value.delete(document.id);
  } else {
    selectedIds.value.add(document.id);
  }

  selectedIds.value = new Set(selectedIds.value);
  emit('select', selectedDocuments.value);
}

function toggleSelectAll(): void {
  if (!props.selectable) return;

  if (isAllSelected.value) {
    selectedIds.value.clear();
  } else {
    filteredDocuments.value.forEach(doc => selectedIds.value.add(doc.id));
  }

  selectedIds.value = new Set(selectedIds.value);
  emit('select', selectedDocuments.value);
}

function handleView(document: Document): void {
  emit('view', document);
}

function handleDownload(document: Document): void {
  emit('download', document);
}

function handleDelete(document: Document): void {
  emit('delete', document);
}

function clearSelection(): void {
  selectedIds.value.clear();
  emit('select', []);
}

// ============================================================================
// Expose
// ============================================================================

defineExpose({
  selectedDocuments,
  clearSelection,
  searchQuery
});
</script>

<template>
  <div class="document-list">
    <!-- Header -->
    <div class="document-list__header">
      <!-- Search -->
      <div v-if="searchable" class="document-list__search">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
          <path d="M15.5 14h-.79l-.28-.27C15.41 12.59 16 11.11 16 9.5 16 5.91 13.09 3 9.5 3S3 5.91 3 9.5 5.91 16 9.5 16c1.61 0 3.09-.59 4.23-1.57l.27.28v.79l5 4.99L20.49 19l-4.99-5zm-6 0C7.01 14 5 11.99 5 9.5S7.01 5 9.5 5 14 7.01 14 9.5 11.99 14 9.5 14z"/>
        </svg>
        <input
          v-model="searchQuery"
          type="text"
          placeholder="Search documents..."
          class="document-list__search-input"
        />
      </div>

      <!-- Selection Info -->
      <div v-if="selectable && selectedIds.size > 0" class="document-list__selection-info">
        <span>{{ selectedIds.size }} selected</span>
        <button class="document-list__clear-btn" @click="clearSelection">
          Clear
        </button>
      </div>

      <!-- Document Count -->
      <div class="document-list__count">
        {{ filteredDocuments.length }} document{{ filteredDocuments.length !== 1 ? 's' : '' }}
      </div>
    </div>

    <!-- Loading State -->
    <div v-if="loading" class="document-list__loading">
      <div class="document-list__spinner"></div>
      <span>Loading documents...</span>
    </div>

    <!-- Empty State -->
    <div v-else-if="filteredDocuments.length === 0" class="document-list__empty">
      <svg width="48" height="48" viewBox="0 0 24 24" fill="currentColor" opacity="0.3">
        <path d="M14 2H6c-1.1 0-2 .9-2 2v16c0 1.1.9 2 2 2h12c1.1 0 2-.9 2-2V8l-6-6zM6 20V4h7v5h5v11H6z"/>
      </svg>
      <p>{{ emptyMessage }}</p>
    </div>

    <!-- Document Table -->
    <div v-else class="document-list__table-container">
      <table class="document-list__table">
        <thead>
          <tr>
            <!-- Checkbox Column -->
            <th v-if="selectable" class="document-list__th--checkbox">
              <input
                type="checkbox"
                :checked="isAllSelected"
                :indeterminate="isSomeSelected"
                @change="toggleSelectAll"
              />
            </th>

            <!-- Name Column -->
            <th
              v-if="columns.includes('name')"
              class="document-list__th--sortable"
              @click="handleSort('name')"
            >
              <span>Name</span>
              <svg
                v-if="sortField === 'name'"
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="currentColor"
                :class="{ 'document-list__sort-icon--desc': sortOrder === 'desc' }"
              >
                <path d="M7 14l5-5 5 5z"/>
              </svg>
            </th>

            <!-- Type Column -->
            <th
              v-if="columns.includes('type')"
              class="document-list__th--sortable"
              @click="handleSort('type')"
            >
              <span>Type</span>
              <svg
                v-if="sortField === 'type'"
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="currentColor"
                :class="{ 'document-list__sort-icon--desc': sortOrder === 'desc' }"
              >
                <path d="M7 14l5-5 5 5z"/>
              </svg>
            </th>

            <!-- Size Column -->
            <th
              v-if="columns.includes('size')"
              class="document-list__th--sortable"
              @click="handleSort('size')"
            >
              <span>Size</span>
              <svg
                v-if="sortField === 'size'"
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="currentColor"
                :class="{ 'document-list__sort-icon--desc': sortOrder === 'desc' }"
              >
                <path d="M7 14l5-5 5 5z"/>
              </svg>
            </th>

            <!-- Date Column -->
            <th
              v-if="columns.includes('date')"
              class="document-list__th--sortable"
              @click="handleSort('createdAt')"
            >
              <span>Date</span>
              <svg
                v-if="sortField === 'createdAt'"
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="currentColor"
                :class="{ 'document-list__sort-icon--desc': sortOrder === 'desc' }"
              >
                <path d="M7 14l5-5 5 5z"/>
              </svg>
            </th>

            <!-- Status Column -->
            <th v-if="columns.includes('status')">Status</th>

            <!-- Actions Column -->
            <th v-if="columns.includes('actions')" class="document-list__th--actions">Actions</th>
          </tr>
        </thead>

        <tbody>
          <tr
            v-for="doc in filteredDocuments"
            :key="doc.id"
            class="document-list__row"
            :class="{ 'document-list__row--selected': selectedIds.has(doc.id) }"
            @click="toggleSelect(doc)"
          >
            <!-- Checkbox -->
            <td v-if="selectable" class="document-list__td--checkbox">
              <input
                type="checkbox"
                :checked="selectedIds.has(doc.id)"
                @click.stop
                @change="toggleSelect(doc)"
              />
            </td>

            <!-- Name -->
            <td v-if="columns.includes('name')" class="document-list__td--name">
              <div class="document-list__name-cell">
                <div
                  class="document-list__icon"
                  :style="{ backgroundColor: getDocumentColor(doc.extension) }"
                >
                  {{ doc.extension.toUpperCase() }}
                </div>
                <div class="document-list__name-info">
                  <span class="document-list__name">{{ doc.name }}</span>
                  <span v-if="doc.pageCount" class="document-list__pages">
                    {{ doc.pageCount }} pages
                  </span>
                </div>
              </div>
            </td>

            <!-- Type -->
            <td v-if="columns.includes('type')" class="document-list__td--type">
              <span class="document-list__type-badge">
                {{ doc.extension.toUpperCase() }}
              </span>
            </td>

            <!-- Size -->
            <td v-if="columns.includes('size')" class="document-list__td--size">
              {{ formatSize(doc.size) }}
            </td>

            <!-- Date -->
            <td v-if="columns.includes('date')" class="document-list__td--date">
              {{ formatDate(doc.createdAt) }}
            </td>

            <!-- Status -->
            <td v-if="columns.includes('status')" class="document-list__td--status">
              <span
                v-if="doc.status"
                class="document-list__status"
                :class="`document-list__status--${doc.status}`"
              >
                {{ doc.status }}
              </span>
            </td>

            <!-- Actions -->
            <td v-if="columns.includes('actions')" class="document-list__td--actions">
              <div class="document-list__actions">
                <button
                  class="document-list__action-btn"
                  @click.stop="handleView(doc)"
                  title="View"
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M12 4.5C7 4.5 2.73 7.61 1 12c1.73 4.39 6 7.5 11 7.5s9.27-3.11 11-7.5c-1.73-4.39-6-7.5-11-7.5zM12 17c-2.76 0-5-2.24-5-5s2.24-5 5-5 5 2.24 5 5-2.24 5-5 5zm0-8c-1.66 0-3 1.34-3 3s1.34 3 3 3 3-1.34 3-3-1.34-3-3-3z"/>
                  </svg>
                </button>
                <button
                  class="document-list__action-btn"
                  @click.stop="handleDownload(doc)"
                  title="Download"
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M19 9h-4V3H9v6H5l7 7 7-7zM5 18v2h14v-2H5z"/>
                  </svg>
                </button>
                <button
                  class="document-list__action-btn document-list__action-btn--danger"
                  @click.stop="handleDelete(doc)"
                  title="Delete"
                >
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor">
                    <path d="M6 19c0 1.1.9 2 2 2h8c1.1 0 2-.9 2-2V7H6v12zM19 4h-3.5l-1-1h-5l-1 1H5v2h14V4z"/>
                  </svg>
                </button>
              </div>
            </td>
          </tr>
        </tbody>
      </table>
    </div>
  </div>
</template>

<style scoped>
.document-list {
  display: flex;
  flex-direction: column;
  background-color: white;
  border-radius: 12px;
  border: 1px solid #e5e7eb;
  overflow: hidden;
}

/* Header */
.document-list__header {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 12px 16px;
  border-bottom: 1px solid #e5e7eb;
  background-color: #f9fafb;
}

.document-list__search {
  display: flex;
  align-items: center;
  gap: 8px;
  flex: 1;
  max-width: 300px;
  padding: 8px 12px;
  background-color: white;
  border: 1px solid #e5e7eb;
  border-radius: 6px;
}

.document-list__search svg {
  color: #9ca3af;
}

.document-list__search-input {
  flex: 1;
  border: none;
  outline: none;
  font-size: 0.875rem;
}

.document-list__selection-info {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 0.875rem;
  color: #3b82f6;
}

.document-list__clear-btn {
  padding: 4px 8px;
  border: none;
  border-radius: 4px;
  background-color: transparent;
  color: #3b82f6;
  font-size: 0.75rem;
  cursor: pointer;
}

.document-list__clear-btn:hover {
  background-color: #eff6ff;
}

.document-list__count {
  margin-left: auto;
  font-size: 0.75rem;
  color: #6b7280;
}

/* Loading & Empty */
.document-list__loading,
.document-list__empty {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 48px;
  color: #6b7280;
}

.document-list__spinner {
  width: 32px;
  height: 32px;
  border: 3px solid #e5e7eb;
  border-top-color: #3b82f6;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
  margin-bottom: 16px;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.document-list__empty p {
  margin: 16px 0 0;
}

/* Table */
.document-list__table-container {
  overflow-x: auto;
}

.document-list__table {
  width: 100%;
  border-collapse: collapse;
}

.document-list__table th,
.document-list__table td {
  padding: 12px 16px;
  text-align: left;
}

.document-list__table th {
  font-size: 0.75rem;
  font-weight: 600;
  color: #6b7280;
  text-transform: uppercase;
  background-color: #f9fafb;
  border-bottom: 1px solid #e5e7eb;
}

.document-list__th--sortable {
  cursor: pointer;
  user-select: none;
}

.document-list__th--sortable:hover {
  color: #374151;
}

.document-list__th--sortable span {
  display: inline-flex;
  align-items: center;
  gap: 4px;
}

.document-list__sort-icon--desc {
  transform: rotate(180deg);
}

.document-list__th--checkbox,
.document-list__td--checkbox {
  width: 40px;
}

.document-list__th--actions,
.document-list__td--actions {
  width: 120px;
}

/* Row */
.document-list__row {
  cursor: pointer;
  transition: background-color 0.2s ease;
}

.document-list__row:hover {
  background-color: #f9fafb;
}

.document-list__row--selected {
  background-color: #eff6ff;
}

.document-list__row td {
  border-bottom: 1px solid #e5e7eb;
}

/* Name Cell */
.document-list__name-cell {
  display: flex;
  align-items: center;
  gap: 12px;
}

.document-list__icon {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border-radius: 6px;
  color: white;
  font-size: 0.625rem;
  font-weight: 700;
  flex-shrink: 0;
}

.document-list__name-info {
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.document-list__name {
  font-size: 0.875rem;
  font-weight: 500;
  color: #374151;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.document-list__pages {
  font-size: 0.75rem;
  color: #6b7280;
}

/* Type Badge */
.document-list__type-badge {
  display: inline-block;
  padding: 2px 8px;
  background-color: #f3f4f6;
  border-radius: 4px;
  font-size: 0.625rem;
  font-weight: 600;
  color: #6b7280;
}

/* Size & Date */
.document-list__td--size,
.document-list__td--date {
  font-size: 0.875rem;
  color: #6b7280;
}

/* Status */
.document-list__status {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 10px;
  font-size: 0.75rem;
  font-weight: 500;
  text-transform: capitalize;
}

.document-list__status--ready {
  background-color: #d1fae5;
  color: #059669;
}

.document-list__status--processing {
  background-color: #fef3c7;
  color: #d97706;
}

.document-list__status--error {
  background-color: #fee2e2;
  color: #dc2626;
}

/* Actions */
.document-list__actions {
  display: flex;
  gap: 4px;
}

.document-list__action-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 28px;
  height: 28px;
  border: none;
  border-radius: 4px;
  background-color: transparent;
  color: #6b7280;
  cursor: pointer;
  transition: all 0.2s ease;
}

.document-list__action-btn:hover {
  background-color: #f3f4f6;
  color: #374151;
}

.document-list__action-btn--danger:hover {
  background-color: #fee2e2;
  color: #ef4444;
}

/* Checkbox styling */
input[type="checkbox"] {
  width: 16px;
  height: 16px;
  cursor: pointer;
}
</style>
