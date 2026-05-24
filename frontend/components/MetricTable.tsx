interface MetricTableProps {
  rows: Array<Record<string, string | number | null>>;
  title: string;
}

export function MetricTable({ rows, title }: MetricTableProps) {
  const columns = rows.length ? Object.keys(rows[0]) : [];
  return (
    <section className="panel">
      <div className="panelHeader">
        <h2>{title}</h2>
        <span>{rows.length} rows</span>
      </div>
      <div className="tableWrap">
        <table>
          <thead>
            <tr>{columns.map((column) => <th key={column}>{column}</th>)}</tr>
          </thead>
          <tbody>
            {rows.slice(0, 25).map((row, index) => (
              <tr key={`${title}-${index}`}>
                {columns.map((column) => <td key={column}>{String(row[column] ?? "")}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}
