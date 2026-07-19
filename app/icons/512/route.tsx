import { ImageResponse } from "next/og";

import { IconArt } from "@/components/icon-art";

export const dynamic = "force-static";

export function GET() {
  return new ImageResponse(<IconArt size={512} />, {
    width: 512,
    height: 512,
  });
}
