function update_table(table, resp, feat) {

    // This functions takes a table id, the server's predicted response and a number of features and loads the response onto the table
    var html = ""
    for (row in resp) {
        students = ["Timmy", "Bob", "Marle", "John"];
        html += "<tr>";
        if (table == "#HTable") {
          html += '<th scope="row" class="text-center">Feature ' + row + '</th>';
        }
        else {
          html += '<th scope="row" class="text-center">' + students[row] + '</th>';
        };
        for (key in resp[row]) {
            html += "<td class='text-center'>" + resp[row][key].toFixed(2) + "</td>";
        };
        html += "</tr>";
    };
    $(table).find('tr').remove();
    $(html).appendTo(table)
    if (table == "#WTable")
        update_head(table, 0, feat);
    else
        update_head(table, 1, 0);

};

function update_head(table, type, features) {

    // This helper function takes a table id, the type of table and a number of features and generates the table head.
    // The tables can be of two types: 1 and else.
    // Tables of type 1 have courses as the head and of type else, features.
    var html = "";
    header = ["Student", "English", "History", "Math"];
    if (type == 1) {
        for (head in header)
            html += '<th class="text-center">' + header[head] + '</th>';
    }
    else {
        html += '<th class="text-center">Student</th>';
        for (var i = 0; i < features; i++)
            html += '<th class="text-center">Feature ' + i + '</th>';
    };
    if (table == "#HTable") {
        $("#HHead").find('th').remove();
        $(html).appendTo("#HHead");
    }
    else if (table == "#WTable") {
        $("#WHead").find('th').remove();
        $(html).appendTo("#WHead");
    }
    else {
        $("#PHead").find('th').remove();
        $(html).appendTo("#PHead");
    };
};
