/**
 * Hybrid search API service
 */

import { apiClient } from './api';
import type { ApiResponse, PaginationParams } from '@/types';

export interface SearchQuery {
  query: string;
  filters?: SearchFilters;
  pagination?: PaginationParams;
  options?: SearchOptions;
}

export interface SearchFilters {
  dateRange?: {
    start: string;
    end: string;
  };
  categories?: string[];
  types?: ('conversation' | 'correction' | 'vocabulary' | 'achievement')[];
  minScore?: number;
}

export interface SearchOptions {
  useSemanticSearch?: boolean;
  useKeywordSearch?: boolean;
  hybridWeight?: number; // 0 = all keyword, 1 = all semantic
  highlightMatches?: boolean;
  includeContext?: boolean;
}

export interface SearchResult {
  id: string;
  type: 'conversation' | 'correction' | 'vocabulary' | 'achievement';
  title: string;
  content: string;
  highlights?: string[];
  score: number;
  metadata: SearchResultMetadata;
  timestamp: string;
}

export interface SearchResultMetadata {
  session_id?: string;
  message_id?: string;
  category?: string;
  context?: string;
}

export interface SearchResponse {
  results: SearchResult[];
  total: number;
  page: number;
  limit: number;
  query: string;
  processingTime: number;
  searchType: 'hybrid' | 'semantic' | 'keyword';
}

export interface SearchSuggestion {
  text: string;
  type: 'recent' | 'popular' | 'related';
  count?: number;
}

/**
 * Perform hybrid search across all content
 */
export const search = async (
  searchQuery: SearchQuery
): Promise<SearchResponse> => {
  const response = await apiClient.post<ApiResponse<SearchResponse>>(
    '/api/search',
    searchQuery
  );

  return response.data.data;
};

/**
 * Get search suggestions based on partial query
 */
export const getSuggestions = async (
  partialQuery: string,
  limit: number = 5
): Promise<SearchSuggestion[]> => {
  const response = await apiClient.get<ApiResponse<SearchSuggestion[]>>(
    '/api/search/suggestions',
    { params: { q: partialQuery, limit } }
  );

  return response.data.data;
};

/**
 * Get recent searches
 */
export const getRecentSearches = async (
  limit: number = 10
): Promise<string[]> => {
  const response = await apiClient.get<ApiResponse<string[]>>(
    '/api/search/recent',
    { params: { limit } }
  );

  return response.data.data;
};

/**
 * Clear recent search history
 */
export const clearRecentSearches = async (): Promise<void> => {
  await apiClient.delete('/api/search/recent');
};

/**
 * Search conversations only
 */
export const searchConversations = async (
  query: string,
  pagination?: PaginationParams
): Promise<SearchResponse> => {
  return search({
    query,
    filters: { types: ['conversation'] },
    pagination,
  });
};

/**
 * Search vocabulary items
 */
export const searchVocabulary = async (
  query: string,
  pagination?: PaginationParams
): Promise<SearchResponse> => {
  return search({
    query,
    filters: { types: ['vocabulary'] },
    pagination,
  });
};

/**
 * Search corrections and learning points
 */
export const searchCorrections = async (
  query: string,
  pagination?: PaginationParams
): Promise<SearchResponse> => {
  return search({
    query,
    filters: { types: ['correction'] },
    pagination,
  });
};

/**
 * Semantic similarity search
 */
export const semanticSearch = async (
  query: string,
  limit: number = 10
): Promise<SearchResult[]> => {
  const response = await search({
    query,
    options: {
      useSemanticSearch: true,
      useKeywordSearch: false,
    },
    pagination: { limit },
  });

  return response.results;
};

export default {
  search,
  getSuggestions,
  getRecentSearches,
  clearRecentSearches,
  searchConversations,
  searchVocabulary,
  searchCorrections,
  semanticSearch,
};
