// src/setupProxy.js

module.exports = function (app) {
    app.use(function (req, res, next) {
        //res.setHeader("Cross-Origin-Opener-Policy", "same-origin")
        //res.setHeader("Cross-Origin-Embedder-Policy", "require-corp")
        //res.setHeader("Cross-Origin-Opener-Policy", "same-origin-allow-popups");
        //res.setHeader("Cross-Origin-Embedder-Policy", "unsafe-none");
        res.setHeader("Cross-Origin-Opener-Policy", "same-origin");
        res.setHeader("Cross-Origin-Embedder-Policy", "credentialless");
        next()
    })
}