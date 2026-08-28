import { defineConfig } from 'vitest/config';

export default defineConfig({
  test: {
    include: ['src/**/*.test.ts'],
    // Several tests run a minute of simulated time; the 5s default fails them
    // on a loaded machine rather than on anything the code did.
    testTimeout: 60_000,
  },
});
