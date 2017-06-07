webswingRequirejs.define(["jquery","text!templates/upload.html","text!templates/upload.css","jquery.iframe-transport","jquery.fileupload"],function(t,n,r){var i=t("<style></style>",{type:"text/css"});return i.text(r),t("head").prepend(i),function(){function x(e,t){a==null&&N(r),a.closest(r.cfg.rootElement).length===0&&r.cfg.rootElement.append(a),f.val(t),C(b,e.allowDownload),C(w,e.allowUpload),C(d,e.allowUpload),C(E,e.allowDelete),S.prop("multiple",e.isMultiSelection),S.attr("accept",e.filter),k(!1),a.show("fast")}function T(){a!=null&&a.closest(r.cfg.rootElement).length!==0&&(a.hide("fast"),a.detach())}function N(){r.cfg.rootElement.append(n),a=r.cfg.rootElement.find('div[data-id="uploadBar"]'),f=a.find('input[data-id="fileDialogTransferBarClientId"]'),l=a.find('div[data-id="fileDialogErrorMessage"]'),c=a.find('div[data-id="fileDialogErrorMessageContent"]'),h=a.find('button[data-id="deleteSelectedButton"]'),p=a.find('button[data-id="downloadSelectedButton"]'),d=a.find('div[data-id="fileDropArea"]'),v=a.find('form[data-id="fileupload"]'),m=a.find('div[data-id="fileDialogTransferProgressBar"]'),g=a.find('div[data-id="progress"] .progress-bar'),y=a.find('div[data-id="cancelBtn"]'),b=a.find('div[data-id="fileDownloadBtn"]'),w=a.find('div[data-id="fileUploadBtn"]'),E=a.find('div[data-id="fileDeleteBtn"]'),S=a.find('input[data-id="fileInput"]'),h.bind("click",function(e){A("deleteFile")}),p.bind("click",function(e){A("downloadFile")}),r.cfg.rootElement.bind("drop",function(e){e.preventDefault()}),r.cfg.rootElement.bind("dragover",function(e){o?clearTimeout(o):d.addClass("in"),o=setTimeout(function(){o=null,d.removeClass("in")},100)});var e=v.fileupload({xhrFields:{withCredentials:!0},url:r.cfg.connectionUrl+"file",dataType:"json",dropZone:d});e.on("fileuploadadd",function(e,t){t.files.forEach(function(e){s.push(e.name)}),i.push(t),k(!0)}),e.bind("fileuploadfail",function(e,t){u?(c.append("<p>"+t.jqXHR.responseText+"</p>"),clearTimeout(o)):(c.append("<p>"+t.jqXHR.responseText+"</p>"),l.show("fast")),u=setTimeout(function(){u=null,c.html(""),l.hide("fast")},5e3)}),e.bind("fileuploadprogressall",function(e,t){var n=parseInt(t.loaded/t.total*100,10);g.css("width",n+"%"),n===100&&(setTimeout(function(){L(s),s=[]},1e3),k(!1),i=[])}),y.click(function(){L([]),i.forEach(function(e){e.abort()}),k(!1)}),a.detach()}function C(e,t){t?e.show():e.hide()}function k(e){e?(g.css("width","0%"),m.show("fast")):(m.hide("fast"),g.css("width","0%"))}function L(e){r.send({uploaded:{files:e}})}function A(e){r.send({events:[{event:{type:e}}]})}function O(e){var t="hiddenDownloader",n=document.getElementById(t);n===null&&(n=document.createElement("iframe"),n.id=t,n.style.display="none",document.body.appendChild(n)),n.src=r.cfg.connectionUrl+e}function M(e){window.open(e,"_blank")}function _(e){window.open(r.cfg.connectionUrl+"print/viewer.html?file="+e,"_blank")}var t=this,r;t.injects=r={cfg:"webswing.config",send:"socket.send"},t.provides={open:x,close:T,download:O,link:M,print:_};var i=[],s=[],o,u,a,f,l,c,h,p,d,v,m,g,y,b,w,E,S}});