export function IconArt({ size }: { size: number }) {
  const inset = Math.round(size * 0.16);
  const dot = Math.round(size * 0.18);
  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        position: "relative",
        alignItems: "center",
        justifyContent: "center",
        overflow: "hidden",
        background: "#11110f",
        color: "#f7f5ef",
      }}
    >
      <div
        style={{
          position: "absolute",
          inset,
          display: "flex",
          border: `${Math.max(2, Math.round(size * 0.012))}px solid rgba(247,245,239,.26)`,
        }}
      />
      <div
        style={{
          width: dot,
          height: dot,
          display: "flex",
          borderRadius: "999px",
          background: "#d9ff43",
          boxShadow: `0 0 0 ${Math.round(size * 0.045)}px rgba(217,255,67,.09)`,
        }}
      />
      <div
        style={{
          position: "absolute",
          bottom: Math.round(size * 0.08),
          display: "flex",
          fontFamily: "Arial, sans-serif",
          fontSize: Math.round(size * 0.065),
          fontWeight: 700,
          letterSpacing: Math.round(size * 0.018),
          textTransform: "uppercase",
        }}
      >
        Lumen
      </div>
    </div>
  );
}
