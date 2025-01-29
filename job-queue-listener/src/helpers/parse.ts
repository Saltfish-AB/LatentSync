export function tryParseInt(value: string, defaultValue: number | null = null): number | null {
  const parsed = parseInt(value, 10);
  return isNaN(parsed) ? defaultValue : parsed;
}
