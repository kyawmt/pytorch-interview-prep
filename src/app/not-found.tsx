import { LinkButton } from "@/components/common/Button";

export default function NotFound() {
  return (
    <div className="card mx-auto max-w-md px-6 py-14 text-center">
      <p className="font-mono text-3xl text-accent">404</p>
      <h1 className="mt-2 text-lg font-semibold text-text">Page not found</h1>
      <p className="mt-1.5 text-sm text-muted">
        That route does not exist. Try the dashboard or search with ⌘K.
      </p>
      <div className="mt-5 flex justify-center gap-2">
        <LinkButton href="/" variant="primary">
          Dashboard
        </LinkButton>
        <LinkButton href="/practice">Practice</LinkButton>
      </div>
    </div>
  );
}
