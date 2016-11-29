function update_table(table, resp) {
    var html = ""
    for (row in resp) {
        html += "<tr>";
        for (key in resp[row]) {
            html += "<td class='text-center'>" + resp[row][key].toFixed(2) + "</td>";
        };
        html += "</tr>";
    };
    $(table).find('tr').remove();
    $(html).appendTo(table)
}
