import type { Metadata } from "next";

import { VT323 } from "next/font/google";
import "./globals.css";

const vt323 = VT323({
  weight: "400",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Felix-Detect-Fix",
  description: "This is the easy to use software to detect bugs and fix them in your code editor for completely free even if in online",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        {/* Favicon link */}
        <link rel="icon" href="/favi.png" sizes="any" />
        {/* Optional PNG favicon */}
        <link rel="icon" type="image/png" href="/favi.png" />
      </head>
      <body className={`${vt323.className} antialiased ml-10 mr-10`}>
        {children}
      </body>
    </html>
  );
}
