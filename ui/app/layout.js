import './globals.css';

export const metadata = {
  title: 'Vietnamese GPT-2 Chat',
  description: 'Simple chat UI for Vietnamese GPT-2 backend'
};

export default function RootLayout({ children }) {
  return (
    <html lang="vi">
      <body>{children}</body>
    </html>
  );
}
