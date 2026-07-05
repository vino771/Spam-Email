const button=document.getElementById("predictBtn");

button.onclick=async()=>{

const message=document.getElementById("message").value;

if(message.trim()==""){

alert("Enter a message.");

return;

}

document.getElementById("loading").style.display="block";

document.getElementById("result").style.display="none";

const response=await fetch("/predict",{

method:"POST",

headers:{

"Content-Type":"application/json"

},

body:JSON.stringify({

message:message

})

});

const data=await response.json();

document.getElementById("loading").style.display="none";

document.getElementById("result").style.display="block";

document.getElementById("prediction").innerHTML=data.prediction;

document.getElementById("confidence").innerHTML="Confidence : "+data.confidence+" %";

if(data.prediction=="Spam"){

document.getElementById("result").className="result spam";

}else{

document.getElementById("result").className="result ham";

}

};