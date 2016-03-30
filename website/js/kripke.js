/**
 * Created by diederik and Siebert and Jos on 13-3-16.
 *
 */

/* global variables */
//Number of cards
var card_value = 0;

//Number of players.
var player_value = 0;

//Cards Drawn.
var card_array = [];

var color_array =['red', 'green', 'blue', 'orange', 'yellow'];

/* Boolean when the probability must be visualize, you dont want it when the whole graph is visualized */
var probability = false;

//Probability Matrix
var probMatrix = null;

//Boolean to keep hold if the game is won
var alreadyWon = false;

//Array to keep track what the other players think that are the gamma(playstyle) of the other players.
var playerGamma =[];

//Array to keep track if the player has called or not
var playerCalls = [];

//These values are used for determining the adjusted probabilities
var playerStyles = [1,1.25,0.75,1,1.5,0.5,1.5];

/* run this at the start to disable the buttons that are not yet needed */
disableButton();

/*Create the graph*/
function playTheGame(viewGraph){
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

    //Empty the tables
    emptyTables();

    //Disable to appropriate buttons.
    disableButton();

    /*Make the graph with a array with all the number in an array */
    if(viewGraph){
        makeFullModel(range1(card_value));
    } else {
        getCard();
        determineKnowledge();
        document.getElementById("draw_card").disabled =true;
    }

    //This is where the call / bluff round is started.
    callBlufRound();
    if(!alreadyWon){
        checkWin();
        alreadyWon= false;
    }
    setUtilityMatrixToArea()
}

/**
 * This will empty the tables.
 */
function emptyTables(){
    document.getElementById("cardArea").innerHTML = "";

    document.getElementById("resultArea").innerHTML = "";

    document.getElementById("bluffArea").innerHTML = "";
}

/**
 *  Disable the person buttons and enable the draw card button
 */
function disableButton(){
    document.getElementById("draw_card").disabled =false;
    document.getElementById("graph").disabled =false;
    document.getElementById("person1").disabled = true;
    document.getElementById("person2").disabled = true;
    document.getElementById("person3").disabled = true;
    document.getElementById("person4").disabled = true;
}

/**
 *  Enable the person buttons
 */
function enableButton(){
    document.getElementById("person1").disabled = false;
    document.getElementById("person2").disabled = false;
    document.getElementById("person3").disabled = false;
    if(player_value==4){
        document.getElementById("person4").disabled = false;
    }
}

/**
 *  Function for calculating the probabilities
 */
function calculateProbability(array) {
    var matrix = [];
    //This is the case where we have the same number of players as of cards.
    if(player_value==card_value){
        var temp_highest = 0;
        var highest_index = 0;
        var i, j = 0;

        for (i=0; i<card_value; i++){
            if(card_array[i]>temp_highest){
                highest_index = i;
                temp_highest = card_array[i];
            }
        }

        for (i=0; i<card_value;i++){
            matrix[i] = [];
            for(j=0;j<card_value;j++){
                if(j==highest_index){
                    matrix[i][j] = 1
                } else {
                    matrix[i][j] =0
                }
            }
        }
    } else {
        for(i=0; i<player_value; i++) {
            matrix[i] = [];
            for(j=0; j<player_value; j++) {
                matrix[i][j] = 0;
            }
        }
        for (i=0;i<array.length;i++){
            var node =array[i];
            var index = 9999;
            for(var q=0; q<card_value;q++){
                if(card_array[q]!=node[q]) {
                    index = q;
                }
            }
            if(index==9999){
                var temp_l =0;
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
    }
    return matrix
}


function setUtilityMatrixToArea(){
    document.getElementById("bluffArea").innerHTML += "";
    var String = "";
    String += "<table class=\"table table-striped\">  <thead> <tr> <td> </td>";

    for (var i=1;i<player_value+1;i++){
        String += " <th> Player " + i + "  (" +playerStyles[i-1]+  ")\t" + " </th> ";
    }
    String += "</tr>";
    for(i=0;i<player_value;i++){

        String += "<tr> <td> <b> Player " + (i+1) + "\t"  + " </b> </td>";
        for(var j=0;j<player_value;j++){
            if(i==j){
                String += " <td> " + "    " + " </td>" ;
            } else {
                String += " <td> (" + playerGamma[j][0].toFixed(2) +")   (" + playerGamma[j][1].toFixed(2)+ ") </td>" ;
            }
        }
        String +=  "</thead> </tr>"
    }
    String += "</table>";
    document.getElementById("bluffArea").innerHTML += String;
}

/* Set the text to the html */
function setTextToResultArea(matrix){
    document.getElementById("resultArea").innerHTML += "";
    var String = "";
    String += "<table class=\"table table-striped\">  <thead> <tr> <td> </td>";


    var i;

    for (i=1;i<player_value+1;i++){
        String += " <th> Player " + i + "  (" +playerStyles[i-1]+  ")\t" + " </th> ";
    }
    String += "</tr>";
    for(i=0;i<player_value;i++){

        String += "<tr> <td> <b> Player " + (i+1) + "\t"  + " </b> </td>";
        for(var j=0;j<player_value;j++){
            if(i==j){
                String += " <td> " + matrix[i][j].toFixed(2)+  "     (" + calculateUtility(matrix[i][j],i).toFixed(2) + ") </td>" ;
            } else {
                String += " <td> " + matrix[i][j].toFixed(2) + " </td>" ;
            }
        }
        String +=  "</thead> </tr>"
    }
    String += "</table>";
    document.getElementById("resultArea").innerHTML += String;
}


function calculateUtility(number, i){
    var gain = player_value -1;
    var utility_win = gain * number;
    var utility_lose = -(1-number);
    var gamma = -(utility_lose/utility_win);
    if(playerGamma[i]==null){
        playerGamma[i] =[-5,5];
    }

        if(playerStyles[i]*utility_win + utility_lose >= 0){
            if(!(gamma>9999)) {
                updateUtility(gamma, true, i);
            }
            playerCalls[i] = 1
        } else {
            updateUtility(gamma,false,i);
            playerCalls[i] = 0
        }

    return (playerStyles[i] * utility_win + utility_lose);
}

function updateUtility(gamma, call, i){
    console.log(playerGamma[i] + "  "  + call + "   " + gamma);
    if(call && playerGamma[i][0]<gamma){
        playerGamma[i][0] = gamma;
    } else if(!(call) && playerGamma[i][1]>gamma){
        playerGamma[i][1] = gamma;
    }
}


/* The possible cases for a person when a card is draw */
function possibleCases(person_number){
    var array = [];

    var i,j = 0;

    for(i=2; i<card_value+2;i++){
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

            for(j=0;j<player_value;j++) {
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

//This will only change one card.
function changeCard(player,card) {
    card_array[player] = parseInt(card);

    var doubleCard = false;
    var i, j;

    for(i=0; i<card_array.length; i++){
        for(j=0; j<card_array.length; j++) {
            if (j != i && card_array[i] == card_array[j]){
                doubleCard = true;
            }
        }
    }

    if(!doubleCard){
        //Empty the tables
        emptyTables();



        var string = "<table class='table table-striped'> <tbody>";
        for(i=1;i<card_array.length+1;i++){
            string += "<tr> <td> Person " + i + " draws card: "+card_array[i-1]+" </td> </tr>";
        }
        string += " </tbody> </table>";

        document.getElementById("cardArea").innerHTML += string;

        enableButton();

        viewAllPersons();

        determineKnowledge();

        //This is where the call / bluff round is started.
        callBlufRound();

        if(!alreadyWon){
            checkWin();
            alreadyWon = false;
        }

        setUtilityMatrixToArea();

    } else {
        //TODO maybe warn the person
    }

}

function removeOptions(selectbox)
{
    var i;

    for(i=selectbox.options.length-1;i>=0;i--)
    {
        selectbox.remove(i);
    }
}

/**
 * This will determien the knowledge for every player.
 */
function determineKnowledge() {
    document.getElementById("knowledgeArea").innerHTML = "";

    var string = "<table id='knowledgetable' class='table table-striped'> <tbody>";

    var i;

    for(i=1;i<card_array.length+1;i++){
        string += "<tr> <td> Person " + i + ":";
        string += "\\begin{align*}";
        string += "M \\models K_" + i + "(";

        //Run through all the other players.,
        for(var c = 1; c < card_array.length+1; c++){

            if( c != i) {
                string += "p_" + c + "c_" + card_array[c-1];
            }

            //Check if we are at the last element or if the last element
            //TODO Haha this should be refactored
            if((c==i &&  c < card_array.length && c != 1) || ( c != i && c < card_array.length && c+1 != i)) {
                string += " \\land "
            }


        }


        if(card_value != player_value) {
            string += ") \\land K_" + i + "(";

            var placed = false;
            //Run through all the other players.,
            for (c = 2; c < card_value + 2; c++) {


                if (card_array.indexOf(c) == -1 || card_array[i-1] == c) {

                    if (placed) {
                        string += " \\lor "
                        placed = false;
                    }

                    string += "p_" + i + "c_" + c;
                    placed = true;
                }

            }

            string += ")";
        } else {
            string +=  " \\land p_" + i + "c_" + card_array[i-1] + ")"
        }

        //string += "\\left[ \\bigwedge_{i=1}^m K_i \\left(\\bigwedge_{j\\neq i} p_j c_{u^{(j)}} \\right) \\right] \\bigwedge_{i=1}^m \\left( K_i \\bigvee_{u^{(i)} \\notin \\{ u^{(j)} \\}_{j\\neq i}} p_i c_{u^{(i)}} \\right)";
        string += "\\end{align*}";
        string += "</td> </tr>";
    }

    string += " </tbody> </table>";

    //Add the table to the card area
    document.getElementById("knowledgeArea").innerHTML += string;

    //Reload Mathjax
    //TODO this is very scary and wrong.
    MathJax.Hub.Queue(["Typeset",MathJax.Hub,"knowledgeArea"]);

}

/*Randomly draw a card for all players and set them to the html */
function getCard() {
    document.getElementById("cardArea").innerHTML = "";
    card_array =[];
    var counter = 0;
    while (counter != player_value) {
        var random_number = Math.floor(Math.random() * card_value) + 2;
        if (!(contains.call(card_array, random_number))) {
            counter = counter + 1;
            card_array.push(random_number);
        }
    }

    enableButton();

    var string = "<table class='table table-striped'> <tbody>";

    var i;

    for(i=1;i<card_array.length+1;i++){
        string += "<tr> <td> Person " + i + " draws card: "+card_array[i-1]+" ";

        removeOptions(document.getElementById("playerCardDropdown"+i+""));
        //string += "<select id=playerCardDropdown"+i+">"

        //Every Option for the card select shoudl be set.
        var select = document.getElementById("playerCardDropdown"+i+"");

        //Run through all the cards, when a card is the same as thecard that the player holds, that will be the selected option.
        for(var c = 2; c < card_value+2; c++){

            var option = document.createElement("option");

            if( c == card_array[i-1]) {

                option.text = c;
                option.value = c;
                option.selected="selected";
                select.appendChild(option);

            } else {

                option.text = c;
                option.value = c;
                select.appendChild(option);

            }

        }

        string += "</td> </tr>";
    }

    string += " </tbody> </table>";

    //Add the table to the card area
    document.getElementById("cardArea").innerHTML += string;

    viewAllPersons();
}

function checkWin(){
    var index = 0;
    var counter = 0;
    for (var k=0;k<player_value;k++){
        if(playerCalls[k]==1){
           index = k;
            counter++;
        }
    }
    if(counter==1){
        var string = "<table class='table table-striped'> <tbody>";
        string += "<tr> <td> Person " + (index+1);
        string += " wins </td> </tr>";
        string += " </tbody> </table>";
        document.getElementById("cardArea").innerHTML += string;
        alreadyWon = true
    }
}


function callBlufRound() {
    //This is where we draw the player choice.
    var string = "<table class='table table-striped'> <tbody>";
    var count_loses = 0;
    for(var k=0; k<player_value; k++) {

        string += "<tr> <td> Person " + (k+1);

        if(playerCalls[k]==1){
            string += " calls </td> </tr>"
        } else {
            if(count_loses==player_value-1){
                string += " wins </td> </tr>";
                alreadyWon = true;
            } else {
                count_loses = count_loses + 1;
                string += " folds </td> </tr>";
            }
        }
    }
    string += " </tbody> </table>";
    document.getElementById("cardArea").innerHTML += string;
}

function viewAllPersons(){
    document.getElementById("resultArea").innerHTML = "";
    var array_possible = [];

    var i;

    for(i=1;i<player_value+1;i++) {
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
        array_number.push(i);
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
            var i, index = -1;

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
        probMatrix = calculateProbability(edges_temp_array);
        setTextToResultArea(probMatrix)
    }

    return edges_array
}

/* Make the graph */
function makeGraph(result){
    var nodes_array = setNodes(result);
    var nodes = new vis.DataSet(nodes_array);

    // create an array with edges
    var edges_array =  setEdges(result);

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

    new vis.Network(container, data, options);
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
