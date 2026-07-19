import NextAuth, { type Session } from "next-auth";
import Google from "next-auth/providers/google";

type GoogleIdentity = {
  email?: unknown;
  email_verified?: unknown;
  sub?: unknown;
};

function allowedGoogleSubjects(): ReadonlySet<string> {
  return new Set(
    (process.env.AUTH_ALLOWED_GOOGLE_SUBS ?? "")
      .split(",")
      .map((subject) => subject.trim())
      .filter(Boolean),
  );
}

export function isAllowedGoogleIdentity(identity: GoogleIdentity): boolean {
  if (identity.email_verified !== true || typeof identity.sub !== "string") {
    return false;
  }

  const subject = identity.sub.trim();
  if (!subject) return false;

  const allowedSubjects = allowedGoogleSubjects();
  if (allowedSubjects.size > 0) return allowedSubjects.has(subject);

  const bootstrapEmail = process.env.AUTH_BOOTSTRAP_EMAIL?.trim().toLowerCase();
  const verifiedEmail =
    typeof identity.email === "string" ? identity.email.trim().toLowerCase() : "";
  return Boolean(bootstrapEmail && verifiedEmail === bootstrapEmail);
}

export const { handlers, auth, signIn, signOut } = NextAuth({
  providers: [
    Google({
      authorization: { params: { prompt: "select_account" } },
    }),
  ],
  session: {
    strategy: "jwt",
    maxAge: 30 * 24 * 60 * 60,
  },
  pages: {
    signIn: "/sign-in",
    error: "/sign-in",
  },
  callbacks: {
    async signIn({ account, profile }) {
      if (account?.provider !== "google" || !profile) return false;

      const identity = profile as GoogleIdentity;
      if (
        typeof identity.sub === "string" &&
        account.providerAccountId !== identity.sub
      ) {
        return false;
      }
      return isAllowedGoogleIdentity(identity);
    },
    async jwt({ token, account, profile }) {
      if (account?.provider === "google") {
        const googleProfile = profile as GoogleIdentity | undefined;
        const profileSubject = googleProfile?.sub;
        token.googleSub =
          typeof profileSubject === "string"
            ? profileSubject
            : account.providerAccountId;
        token.googleEmail =
          typeof googleProfile?.email === "string"
            ? googleProfile.email
            : token.email;
        token.googleEmailVerified = googleProfile?.email_verified === true;
      }

      if (
        !isAllowedGoogleIdentity({
          sub: token.googleSub,
          email: token.googleEmail,
          email_verified: token.googleEmailVerified,
        })
      ) {
        delete token.googleSub;
        delete token.googleEmail;
        delete token.googleEmailVerified;
      }
      return token;
    },
    async session({ session, token }) {
      if (session.user && typeof token.googleSub === "string") {
        session.user.id = token.googleSub;
      }
      return session;
    },
    authorized({ auth: session }) {
      return Boolean(session?.user?.id);
    },
  },
});

export class UnauthenticatedError extends Error {
  constructor() {
    super("Authentication required");
    this.name = "UnauthenticatedError";
  }
}

export async function requireUserId(): Promise<string> {
  const session = await auth();
  if (!session?.user?.id) throw new UnauthenticatedError();
  return session.user.id;
}

export async function requireUser(): Promise<Session["user"] & { id: string }> {
  const session = await auth();
  if (!session?.user?.id) throw new UnauthenticatedError();
  return session.user;
}
