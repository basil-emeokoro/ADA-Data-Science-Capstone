import { useEffect, useState } from "react";

export function useTheme() {
  const [theme, setTheme] = useState<"light" | "dark">("light");

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  return {
    theme,
    toggleTheme: () => setTheme((value) => (value === "light" ? "dark" : "light"))
  };
}
