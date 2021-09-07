
function loading_submit(){
    var article_body = document.getElementById('article-body').value;
    var headline = document.getElementById('headline').value;

    if((headline.length!=0 || headline) && (article_body.length!=0 || article_body)){
        document.getElementById("content").style.display = 'none';
        document.getElementById("loading").style.display = 'block';  
    }    
}

function loading_random(){

        document.getElementById("content").style.display = 'none';
        document.getElementById("loading").style.display = 'block';  
  
}

// Example starter JavaScript for disabling form submissions if there are invalid fields
function validation() {
    'use strict'
  
    // Fetch all the forms we want to apply custom Bootstrap validation styles to
    var forms = document.querySelectorAll('.needs-validation')
  
    // Loop over them and prevent submission
    Array.prototype.slice.call(forms)
      .forEach(function (form) {
        form.addEventListener('submit', function (event) {
        var headline = document.getElementById('headline').value;
        var article_body = document.getElementById('article-body').value;
          if (!form.checkValidity() || ((headline.length==0 || !headline) && (article_body.length==0 || !article_body))) {
            event.preventDefault()
            event.stopPropagation()
          }
  
          form.classList.add('was-validated')
        }, false)
      })
  }



$(document).ready(function(){
  var checkbox_status = $("#checkbox-status").data("checked");
  if (checkbox_status == "False"){
      $("#flexCheckChecked").attr("checked",false);
      $('#checkBoxHidden').val('false')
  }else{
    $("#flexCheckChecked").attr("checked",true);
    $('#checkBoxHidden').val('true')
  }
});

function checkboxChanged(){

    if($('#flexCheckChecked').is(":checked")) {
        $('#checkBoxHidden').val('true')
    }else{
      $('#checkBoxHidden').val('false')
    }   
  }


  function sort_cards_by_date(){
    var card_board = $('#cards')

    cards = card_board.children('div').detach().get()


    cards.sort(function(a, b) {
      date_1 = Date.parse($(a).find("#date").text())
      date_2 = Date.parse($(b).find("#date").text())
      return new Date(date_2 - date_1);
    });
    


    card_board.append(cards); // <== switched this to append

    // console.log()

  }
  
  function sort_cards_by_similarity(){
    var card_board = $('#cards')

    cards = card_board.children('div').detach().get()


    cards.sort(function(a, b) {
      score_1 = $(a).find("#similarity").text().replace("% similar","")
      score_2 = $(b).find("#similarity").text().replace("% similar","")
      // console.log(score_1)
      return score_2 - score_1
    });
    


    card_board.append(cards); // <== switched this to append

    // console.log()

  }





