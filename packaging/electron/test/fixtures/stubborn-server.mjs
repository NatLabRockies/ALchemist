// Ignores stdin closing: stands in for a server that won't shut down gracefully.
process.stdin.resume()
process.stdin.on('end', () => {})
setInterval(() => {}, 1000)
