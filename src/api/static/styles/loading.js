function loading(){
    document.getElementById("content").style.display = 'none';
    document.getElementById("loading").style.display = 'block';      
}

// Example starter JavaScript for disabling form submissions if there are invalid fields
(function () {
    'use strict'
  
    // Fetch all the forms we want to apply custom Bootstrap validation styles to
    var forms = document.querySelectorAll('.needs-validation')
  
    // Loop over them and prevent submission
    Array.prototype.slice.call(forms)
      .forEach(function (form) {
        form.addEventListener('submit', function (event) {
            var headline = $('#headline');
            var content = $('#article-body');
          if (!form.checkValidity() | headline.length<2 | content.length<2) {
            event.preventDefault()
            event.stopPropagation()
          }
          
          form.classList.add('was-validated')
        }, false)
      })
  })()
    