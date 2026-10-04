/** @type {import('next').NextConfig} */
const nextConfig = {
  turbopack: {
    rules: {}
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:5080/api/:path*"
      },
      {
        source: "/ai/:path*",
        destination: "http://localhost:5090/ai/:path*"
      }
    ];
  }
};

export default nextConfig;