import { redirect } from "next/navigation";

import { auth } from "@/auth";
import { AccountMenu } from "@/components/account-menu";
import { LumenApp } from "@/components/lumen-app";

export const dynamic = "force-dynamic";

export default async function HomePage() {
  const session = await auth();
  if (!session?.user?.id) redirect("/sign-in");

  return (
    <LumenApp
      accountMenu={
        <AccountMenu name={session.user.name} email={session.user.email} />
      }
    />
  );
}
