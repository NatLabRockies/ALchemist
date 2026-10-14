import { toast as sonnerToast, type ExternalToast } from 'sonner';

/**
 * How long error and warning toasts stay up. Sonner's 4 s default is too short
 * to read a long API error (e.g. a DesignNotEstimableError naming its terms).
 * Success and info toasts keep the default. A caller's own `duration` wins.
 */
export const PROBLEM_TOAST_DURATION_MS = 10_000;

type Message = Parameters<typeof sonnerToast.error>[0];

/** Sonner's `toast`, with errors and warnings held for PROBLEM_TOAST_DURATION_MS. */
export const toast = Object.assign(
  (message: Parameters<typeof sonnerToast>[0], data?: ExternalToast) => sonnerToast(message, data),
  sonnerToast,
  {
    error: (message: Message, data?: ExternalToast) =>
      sonnerToast.error(message, { duration: PROBLEM_TOAST_DURATION_MS, ...data }),
    warning: (message: Message, data?: ExternalToast) =>
      sonnerToast.warning(message, { duration: PROBLEM_TOAST_DURATION_MS, ...data }),
  },
);
