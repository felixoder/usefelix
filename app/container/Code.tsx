"use client";
import { useState } from "react";
import SyntaxHighlighter from "react-syntax-highlighter";
import { nightOwl } from "react-syntax-highlighter/dist/esm/styles/hljs";
import { Clipboard } from "lucide-react"; // Import copy icon

export default function Code({ children }: { children: React.ReactNode }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    const textToCopy =
      typeof children === "string" ? children : String(children);
    navigator.clipboard.writeText(textToCopy);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500); // Reset "Copied" message after 1.5 sec
  };

  return (
    <div className="relative mt-2 mb-2 text-xs group">
      {/* Copy Button */}
      <button
        onClick={handleCopy}
        className="absolute top-2 right-2 p-1 bg-gray-700 text-white rounded-md opacity-0 group-hover:opacity-100 transition"
      >
        {copied ? "✅ Copied" : <Clipboard size={16} />}
      </button>

      {/* Code Block */}
      <SyntaxHighlighter
        language="python"
        style={nightOwl}
        className="rounded-lg cursor-pointer"
      >
        {typeof children === "string" ? children : String(children)}
      </SyntaxHighlighter>
    </div>
  );
}
