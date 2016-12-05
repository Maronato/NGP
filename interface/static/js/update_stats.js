function resize_cp(plot_id, table) {
    var v_width = $(window).width();
    var plot = "#plot" + plot_id;
    var div = "#plotdiv" + plot_id;

    if (v_width > 480) {
        var size = 1 / 3 * v_width;
        $(div).attr("class", "col-xs-offset-4")
    } else if (v_width < 480) {
        var size = v_width;
        $(div).attr("class", "")
    };

    $(plot).empty();
    $(plot).attr("width", size);
    $(plot).attr("height", size);
    circ_pack(plot, table);
};

//http://stackoverflow.com/questions/2854407/javascript-jquery-window-resize-how-to-fire-after-the-resize-is-completed
var waitForFinalEvent = (function() {
    var timers = {};
    return function(callback, ms, uniqueId) {
        if (!uniqueId) {
            uniqueId = "Don't call this twice without a uniqueId";
        }
        if (timers[uniqueId]) {
            clearTimeout(timers[uniqueId]);
        }
        timers[uniqueId] = setTimeout(callback, ms);
    };
})();

$(window).resize(function() {

    waitForFinalEvent(function() {
        res_all();
    }, 500, "plotresize");
});
