export { auth as proxy } from "@/auth";

export const config = {
  matcher: [
    "/((?!api|sign-in|_next/static|_next/image|icon|apple-icon|manifest.webmanifest|robots.txt|sw\\.js).*)",
  ],
};
