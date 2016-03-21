/**
 * Created by diederik on 13-3-16.
 */

/* global variables */
card_value = 0;
player_value = 0;
card_array = [];
color_array =['red', 'green', 'blue', 'orange', 'yellow']
/* Boolean when the probability must be visualize, you dont want it when the whole graph is visualized */
probability = false;
/* run this at the start to disable the button that are not yet needed */
disableButton();

/*Create the graph*/
function createGraph(viewGraph){
    var options = document.getElementById("dropdown");
    var dropdown_value = options.options[options.selectedIndex].value;
    /* Control statement*/
    if(!(dropdown_value=="34"||dropdown_value=="35"||dropdown_value=="36"||dropdown_value=="45")) {
        viewGraph = false;
    }

    /* get the second value of the drop down menu */
    card_value = parseInt(dropdown_value[1]);
    probability = false;
    /* get the first value of the drop down menu */
    player_value = parseInt(dropdown_value[0]);
    document.getElementById("cardArea").innerHTML = "";

    document.getElementById("resultArea").innerHTML = "";
    disableButton()
    /*Make the graph with a array with all the number in an array */
    if(viewGraph){
        makeFullModel(range1(card_value))
    } else {
        getCard()
        document.getElementById("draw_card").disabled =true;
    }
}

/* Disable the person buttons and enable the draw card button */
function disableButton(){
    document.getElementById("draw_card").disabled =false;
    document.getElementById("graph").disabled =false;
    document.getElementById("person1").disabled = true;
    document.getElementById("person2").disabled = true;
    document.getElementById("person3").disabled = true;
    document.getElementById("person4").disabled = true;
}

/* Enable the person buttons */
function enableButton(){

    document.getElementById("person1").disabled = false;
    document.getElementById("person2").disabled = false;
    document.getElementById("person3").disabled = false;
    if(player_value==4){
        document.getElementById("person4").disabled = false;
    }
}


/* Function to set the text to the html */
function setTextToCardArea(text){
    document.getElementById("cardArea").innerHTML += text;
}


/* Function for calculating the probabilities */
function calculateProbability(array) {
    var matrix = [];
    for(var i=0; i<player_value; i++) {
        matrix[i] = [];
        for(var j=0; j<player_value; j++) {
            matrix[i][j] = 0;
        }
    }
    for (var i=0;i<array.length;i++){
        var node =array[i];
        var index = 9999
        for(var q=0; q<card_value;q++){
            if(card_array[q]!=node[q]) {
                index = q;
            }
        }
        if(index==9999){
            var temp_l =0
            for(var l=0;l<node.length;l++){
                if(node[temp_l]<node[l]){
                    temp_l = l
                }
            }
            for(q=0;q<player_value;q++){
                matrix[q][temp_l] = matrix[q][temp_l] + (1/((card_value-player_value)+1));
            }
        } else{
            temp_l=0;
            for(l=0;l<node.length;l++){
                if(node[temp_l]<node[l]){
                    temp_l = l
                }
            }
            matrix[index][temp_l] = matrix[index][temp_l] + (1/((card_value-player_value)+1));
        }
    }
    return matrix
}


/* Set the text to the html */
function setTextToResultArea(matrix){
    document.getElementById("resultArea").innerHTML += "";
    var String = "";
    String += "<table class=\"table table-striped\">  <thead> <tr> <td> </td>";

    for (var i=1;i<player_value+1;i++){
        String += " <th> Player " + i + "\t" + " </th> ";
    }
    String += "</tr>";
    for(i=0;i<player_value;i++){

        String += "<tr> <td> <b> Player " + (i+1) + "\t"  + " </b> </td>"
        for(var j=0;j<player_value;j++){
            String += " <td> " + matrix[i][j] + " </td>" ;
        }
        String +=  "</thead> </tr>"
    }
    String += "</table>";
    document.getElementById("resultArea").innerHTML += String;
}

/* The possible cases for a person when a card is draw */
function possibleCases(person_number){
    var array = [];

    for(var i=2; i<card_value+2;i++){
        var already_in = false;
        for(var k=0; k<player_value;k++){
            if(i==card_array[k] && k!=person_number-1){
                already_in = true;
                break;
            }
        }
        if(!already_in){
            var temp_card_array = card_array.slice();
            temp_card_array[person_number-1] = i;
            var string = "";

            for(var j=0;j<player_value;j++) {
                string = string + temp_card_array[j];
            }

            array.push(string);
        }
    }
    return array;
}

/* Make the graph for only one person, where number is the number of the person */
function viewPerson(number){
    /* Get all possible cases for a single person */
    probability = false;
    var result = possibleCases(number);
    /* Get the nodes and set them to an array */
    makeGraph(result)
}

/*Randomly draw a card for all players and set them to the html */
function getCard() {
    document.getElementById("cardArea").innerHTML = "";
    card_array =[]
    var counter = 0;
    while (counter != player_value) {
        var random_number = Math.floor(Math.random() * card_value) + 2;
        if (!(contains.call(card_array, random_number))) {
            counter = counter + 1;
            card_array.push(random_number);
        }
    }
    enableButton();

    string = "<table class='table table-striped'> <tbody>";


    for(var i=1;i<card_array.length+1;i++){
        string += "<tr> <td> Person " + i + " draws card " + card_array[i-1] + "</td> </tr>";
    }

    string += " </tbody> </table>"

    setTextToCardArea(string)

    viewAllPersons()
}

function viewAllPersons(){
    document.getElementById("resultArea").innerHTML = "";
    var array_possible = []
    for(var i=1;i<player_value+1;i++) {
        var temp_array = possibleCases(i);
        for(var j=0;j<temp_array.length;j++){
            if (!(contains.call(array_possible, temp_array[j]))) {
                array_possible.push(temp_array[j])
            }
        }
    }
    probability = true;
    makeGraph(array_possible)
}

/*Generate recursive all possible cases for all cards and person */
/*TODO: I think this is the bottleneck why the program is slow */
function allPossibleCases(arr) {
    if (arr.length == 1) {
        return arr[0];
    } else {
        var result = [];
        var allCasesOfRest = allPossibleCases(arr.slice(1));  // recur with the rest of array
        for (var i = 0; i < allCasesOfRest.length; i++) {
            for (var j = 0; j < arr[0].length; j++) {
                for (var k = 0; k < allCasesOfRest.length; k++) {
                    /*Check if a symbol is already in the array*/
                    if (arr[0][j] == allCasesOfRest[i][k]) {
                        break;
                    }
                    /*If it is a new symbol then put it in the array */
                    if (k == allCasesOfRest.length - 1 && arr[0][j] != allCasesOfRest[i]) {
                        result.push(allCasesOfRest[i].toString() + arr[0][j].toString());
                    }
                }
            }
        }
    }
    /*Result should be an array with all different card value like: [5,4,3] for 3 person and 4 card value*/
    return result;
}

/*Make an array for all number until the variable number, but starts from 2*/
function range1(number){
    var array_number =[];
    for (var i=2;i<number+2;i++){
        array_number.push(i)
    }
    return array_number;
}


/*It search if a character is in the string already so for example if 'a' is in 'abc' */
var contains = function(needle) {
    // Per spec, the way to identify NaN is that it is not equal to itself
    var findNaN = needle !== needle;
    var indexOf;

    if(!findNaN && typeof Array.prototype.indexOf === 'function') {
        indexOf = Array.prototype.indexOf;
    } else {
        indexOf = function(needle) {
            var i = -1, index = -1;

            for(i = 0; i < this.length; i++) {
                var item = this[i];

                if((findNaN && item !== item) || item === needle) {
                    index = i;
                    break;
                }
            }

            return index;
        };
    }

    return indexOf.call(this, needle) > -1;
};

/*Set the nodes to an array that is in the form of the node for the vis dataset*/
function setNodes(result){
    var nodes_array = [];
    for (var i = 0; i < result.length; i++) {
        nodes_array.push({id: i, label: result[i]})
    }
    return nodes_array
}

/*Set the edges to an array, that is in the form of the edge for the vis dataset, it will only set an edge
 when there are two character are equal so for example it will set an edge between 234 and 334*/
function setEdges(result){
    var edges_array = [];
    var edges_temp_array=[];
    for (var i = 0; i < result.length; i++) {
        for (var j = 0; j < result.length; j++) {
            var counter = 0;
            var index = 0;
            if (i != j) {
                for (var k = 0; k < player_value; k++) {
                    if (result[i][k] != result[j][k]) {
                        counter = counter + 1;
                        index = k
                    }
                    if (counter == 1 && k == player_value - 1) {
                        if(probability) {
                            if (!(contains.call(edges_temp_array, result[i]))) {
                                edges_temp_array.push(result[i])
                            }
                            if (!(contains.call(edges_temp_array, result[j]))) {
                                edges_temp_array.push(result[j])
                            }
                        }
                        edges_array.push({from: i, to: j, arrows: 'to', label: 'R' + (index+1), color: color_array[index]});
                    }
                }
            }
        }
    }
    if(probability){
        var matrix = calculateProbability(edges_temp_array)
        setTextToResultArea(matrix)
    }
    return edges_array
}

/* Make the graph */
function makeGraph(result){
    var nodes_array = setNodes(result);
    var nodes = new vis.DataSet(nodes_array);

    // create an array with edges
    var edges_array =  setEdges(result)

    var edges = new vis.DataSet(edges_array);
    // create a network
    var container = document.getElementById('mynetwork');
    var data = {
        nodes: nodes,
        edges: edges
    };
    var options = {
        layout:{
            improvedLayout:false
        }
    };
    var network = new vis.Network(container, data, options);
}

/*Make the big graph, where are all possibles cases can be modelled */
function makeFullModel(range_array){

    var array = [];
    for (var i = 0; i < player_value; i++) {
        array[i] = range_array;
    }
    var result = allPossibleCases(array);

    makeGraph(result)
}
