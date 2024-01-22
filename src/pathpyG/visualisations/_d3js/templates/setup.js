// Load via requireJS if available (jupyter notebook environment)
try {
    // Problem: require.config will raise an exception when called for the second time 
    require.config({
        paths: {
            d3: "$d3js".replace(".js", "")
        }
    });
    console.log("OKAY: requireJS was detected.");
}
catch(err){
    // a reference error indicates that requireJS does not exist. 
    // other errors may occur due to multiple calls to config
    if (err instanceof ReferenceError){
        console.log("WARNING: NO requireJS was detected!");

        // Helper function that waits for d3js to be loaded
        require = function require(symbols, callback) {
            var ms = 10;
            window.setTimeout(function(t) {
                if (window[symbols[0]])
                    callback(window[symbols[0]]);
                else 
                    window.setTimeout(arguments.callee, ms);
            }, ms);
        }
    }
};
