import { signOut } from "@/auth";

type AccountMenuProps = {
  email?: string | null;
  name?: string | null;
};

function initials(name?: string | null, email?: string | null): string {
  const source = name?.trim() || email?.trim() || "L";
  const parts = source.split(/[\s@]+/).filter(Boolean);
  return parts
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase())
    .join("");
}

export function AccountMenu({ email, name }: AccountMenuProps) {
  return (
    <details className="account-menu">
      <summary aria-label="Open account menu">
        <span className="account-avatar" aria-hidden="true">
          {initials(name, email)}
        </span>
      </summary>
      <div className="account-popover">
        <p className="eyebrow">Signed in</p>
        <strong>{name || "Lumen curator"}</strong>
        {email ? <small>{email}</small> : null}
        <form
          action={async () => {
            "use server";
            await signOut({ redirectTo: "/sign-in" });
          }}
        >
          <button className="text-button" type="submit">
            Sign out
          </button>
        </form>
      </div>
    </details>
  );
}
