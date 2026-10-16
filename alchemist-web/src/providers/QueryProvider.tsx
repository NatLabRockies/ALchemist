/**
 * React Query Provider
 * Wraps the app with TanStack Query for server state management
 */
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import type { ReactNode } from 'react';

// Create a client
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Stale time: how long data is considered fresh (5 minutes)
      staleTime: 5 * 60 * 1000,
      // Cache time: how long inactive data stays in cache (10 minutes)
      gcTime: 10 * 60 * 1000,
      // Retry failed requests
      retry: 1,
      // Refetch on window focus (useful for keeping session data fresh)
      refetchOnWindowFocus: false,
    },
    mutations: {
      // Never retry a mutation: a failed train, suggest, CSV upload or batch
      // add must not be replayed (a half-applied one would run twice)
      retry: 0,
    },
  },
});

interface QueryProviderProps {
  children: ReactNode;
}

/**
 * Provider component that wraps the app
 */
export function QueryProvider({ children }: QueryProviderProps) {
  return (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
}

export { queryClient };
