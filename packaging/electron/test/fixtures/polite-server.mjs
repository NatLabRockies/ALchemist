// Exits cleanly when stdin closes, as alchemist-server --watch-stdin does.
process.stdin.resume()
process.stdin.on('end', () => process.exit(0))
setInterval(() => {}, 1000)
