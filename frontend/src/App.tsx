import { apiUrl } from "./api";

export function App() {
  return (
    <main data-api-url={apiUrl}>
      <h1>Resume Reader</h1>
      <p>App shell loaded.</p>
    </main>
  );
}
