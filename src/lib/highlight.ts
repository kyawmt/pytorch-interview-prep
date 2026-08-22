/**
 * Tiny dependency-free Python tokenizer used by <CodeBlock> and the editor
 * overlay. Monaco/Shiki would add hundreds of kilobytes for snippets that are
 * rarely longer than 15 lines, so we tokenize with one ordered regex pass.
 */

export type TokenKind =
  | "comment"
  | "string"
  | "number"
  | "keyword"
  | "builtin"
  | "decorator"
  | "self"
  | "function"
  | "attribute"
  | "operator"
  | "punctuation"
  | "blank"
  | "plain";

export interface Token {
  kind: TokenKind;
  value: string;
}

const KEYWORDS = new Set([
  "and", "as", "assert", "async", "await", "break", "class", "continue", "def",
  "del", "elif", "else", "except", "finally", "for", "from", "global", "if",
  "import", "in", "is", "lambda", "nonlocal", "not", "or", "pass", "raise",
  "return", "try", "while", "with", "yield",
]);

const CONSTANTS = new Set(["True", "False", "None"]);

const BUILTINS = new Set([
  "print", "len", "range", "int", "float", "str", "bool", "list", "dict", "set",
  "tuple", "enumerate", "zip", "super", "isinstance", "sum", "min", "max",
  "abs", "sorted", "type", "open",
]);

/**
 * Ordered alternation — first match wins, which is what keeps strings and
 * comments from being re-tokenized as code.
 */
const TOKEN_RE = new RegExp(
  [
    /(?<comment>#[^\n]*)/.source,
    /(?<string>[rbfu]{0,2}("""[\s\S]*?"""|'''[\s\S]*?'''|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])*'))/.source,
    /(?<decorator>@[A-Za-z_][\w.]*)/.source,
    /(?<number>\b\d[\d_]*\.?\d*(?:[eE][-+]?\d+)?\b|\b0[xX][\da-fA-F]+\b)/.source,
    /(?<name>[A-Za-z_]\w*)/.source,
    /(?<operator>\*\*|\/\/|[-+*/%@<>=!&|^~]=?|:=)/.source,
    /(?<punctuation>[()[\]{}:,.;])/.source,
    /(?<space>\s+)/.source,
  ].join("|"),
  "g",
);

export function tokenizePython(code: string): Token[] {
  const tokens: Token[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  TOKEN_RE.lastIndex = 0;

  const push = (kind: TokenKind, value: string) => {
    if (!value) return;
    const previous = tokens[tokens.length - 1];
    if (previous && previous.kind === kind) previous.value += value;
    else tokens.push({ kind, value });
  };

  while ((match = TOKEN_RE.exec(code))) {
    if (match.index > lastIndex) push("plain", code.slice(lastIndex, match.index));
    lastIndex = match.index + match[0].length;
    const groups = match.groups ?? {};

    if (groups.comment) push("comment", groups.comment);
    else if (groups.string) push("string", groups.string);
    else if (groups.decorator) push("decorator", groups.decorator);
    else if (groups.number) push("number", groups.number);
    else if (groups.name) {
      const name = groups.name;
      const before = code.slice(0, match.index);
      const after = code.slice(lastIndex);
      const isCall = /^\s*\(/.test(after);
      const isAttribute = /\.\s*$/.test(before);
      if (KEYWORDS.has(name)) push("keyword", name);
      else if (CONSTANTS.has(name)) push("builtin", name);
      else if (name === "self" || name === "cls") push("self", name);
      else if (isCall) push("function", name);
      else if (BUILTINS.has(name)) push("builtin", name);
      else if (isAttribute) push("attribute", name);
      else push("plain", name);
    } else if (groups.operator) push("operator", groups.operator);
    else if (groups.punctuation) push("punctuation", groups.punctuation);
    else push("plain", match[0]);
  }

  if (lastIndex < code.length) push("plain", code.slice(lastIndex));
  return tokens;
}

export const TOKEN_CLASS: Record<TokenKind, string> = {
  comment: "text-[var(--tok-comment)] italic",
  string: "text-[var(--tok-string)]",
  number: "text-[var(--tok-number)]",
  keyword: "text-[var(--tok-keyword)]",
  builtin: "text-[var(--tok-builtin)]",
  decorator: "text-[var(--tok-decorator)]",
  self: "text-[var(--tok-self)] italic",
  function: "text-[var(--tok-function)]",
  attribute: "text-[var(--tok-attribute)]",
  operator: "text-[var(--tok-operator)]",
  punctuation: "text-[var(--tok-punctuation)]",
  blank: "",
  plain: "text-[var(--tok-plain)]",
};
