import type { MetadataRoute } from "next";

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: "Lumen — Photography Ranker",
    short_name: "Lumen",
    description: "A private photography taste engine.",
    id: "/",
    start_url: "/",
    scope: "/",
    display: "standalone",
    orientation: "any",
    background_color: "#11110f",
    theme_color: "#11110f",
    categories: ["photo", "lifestyle"],
    icons: [
      { src: "/icons/192", sizes: "192x192", type: "image/png" },
      { src: "/icons/512", sizes: "512x512", type: "image/png" },
      {
        src: "/icons/512",
        sizes: "512x512",
        type: "image/png",
        purpose: "maskable",
      },
    ],
    shortcuts: [
      { name: "Rank photographs", short_name: "Rank", url: "/#rank" },
      {
        name: "View collection",
        short_name: "Collection",
        url: "/#collection",
      },
    ],
  };
}
