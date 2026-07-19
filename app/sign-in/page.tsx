import { redirect } from "next/navigation";

import { auth, signIn } from "@/auth";

const ERROR_COPY: Record<string, string> = {
  AccessDenied: "That Google account is not on this private collection’s allowlist.",
  Configuration: "Google sign-in is not configured yet. Check the Vercel environment variables.",
  OAuthAccountNotLinked: "Use the Google account originally connected to this collection.",
  OAuthCallbackError: "Google could not complete sign-in. Please try once more.",
};

type SignInPageProps = {
  searchParams: Promise<{ error?: string }>;
};

export default async function SignInPage({ searchParams }: SignInPageProps) {
  const session = await auth();
  if (session?.user?.id) redirect("/");

  const { error } = await searchParams;
  const errorMessage = error
    ? ERROR_COPY[error] ?? "Sign-in could not be completed. Please try again."
    : null;

  return (
    <main className="sign-in-page">
      <div className="sign-in-art" aria-hidden="true">
        <span className="art-frame art-frame-one" />
        <span className="art-frame art-frame-two" />
        <span className="art-frame art-frame-three" />
        <span className="art-caption">Light, place, memory.</span>
      </div>

      <section className="sign-in-panel" aria-labelledby="sign-in-title">
        <div className="sign-in-brand">
          <span className="brand-mark" aria-hidden="true" />
          <span>Lumen</span>
        </div>
        <div className="sign-in-copy">
          <p className="eyebrow">Your private photography canon</p>
          <h1 id="sign-in-title">
            Find the images<br />
            that <em>stay.</em>
          </h1>
          <p>
            Train a taste model one choice at a time, then let it search for the
            photographs you have not found yet.
          </p>
        </div>

        {errorMessage ? (
          <p className="sign-in-error" role="alert">
            {errorMessage}
          </p>
        ) : null}

        <form
          action={async () => {
            "use server";
            await signIn("google", { redirectTo: "/" });
          }}
        >
          <button className="google-button" type="submit">
            <svg viewBox="0 0 24 24" aria-hidden="true">
              <path fill="#4285F4" d="M21.6 12.23c0-.71-.06-1.39-.18-2.05H12v3.87h5.38a4.6 4.6 0 0 1-2 3.02v2.51h3.24c1.9-1.75 2.98-4.33 2.98-7.35Z" />
              <path fill="#34A853" d="M12 22c2.7 0 4.98-.9 6.63-2.42l-3.24-2.51c-.9.6-2.05.96-3.39.96-2.61 0-4.82-1.76-5.61-4.13H3.04v2.59A10 10 0 0 0 12 22Z" />
              <path fill="#FBBC05" d="M6.39 13.9A6 6 0 0 1 6.08 12c0-.66.11-1.3.31-1.9V7.51H3.04A10 10 0 0 0 2 12c0 1.61.39 3.14 1.04 4.49l3.35-2.59Z" />
              <path fill="#EA4335" d="M12 5.97c1.47 0 2.79.5 3.83 1.5l2.87-2.88A9.62 9.62 0 0 0 12 2a10 10 0 0 0-8.96 5.51l3.35 2.59C7.18 7.73 9.39 5.97 12 5.97Z" />
            </svg>
            Continue with Google
          </button>
        </form>
        <p className="sign-in-privacy">
          <span aria-hidden="true">●</span> Only the verified owner account can
          enter. Your collection stays private.
        </p>
      </section>
    </main>
  );
}
