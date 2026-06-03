/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // three.js ships untranspiled ESM with some addons that Next's server
  // compiler chokes on unless we transpile them explicitly.
  transpilePackages: ["three"],
};

export default nextConfig;
