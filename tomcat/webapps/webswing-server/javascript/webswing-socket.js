webswingRequirejs.define(["atmosphere","ProtoBuf","text!webswing.proto"],function(t,n,r){var i=n.loadProto(r,"webswing.proto"),s=i.build("org.webswing.server.model.proto.InputEventsFrameMsgInProto"),o=i.build("org.webswing.server.model.proto.AppFrameMsgOutProto");return function(){function l(){var e={url:r.cfg.connectionUrl+"async/swing",contentType:"application/json",transport:"websocket",trackMessageLength:!0,reconnectInterval:5e3,fallbackTransport:"long-polling",enableXDR:!0,headers:{}};a&&(e.url=e.url+"-bin",e.headers["X-Atmosphere-Binary"]=!0,e.enableProtocol=!1,e.trackMessageLength=!1,e.contentType="application/octet-stream",e.webSocketBinaryType="arraybuffer"),r.cfg.recordingPlayback&&(e.url=r.cfg.connectionUrl+"async/swing-play",e.headers.file=r.cfg.recordingPlayback),r.cfg.args!=null&&(e.headers["X-webswing-args"]=r.cfg.args),r.cfg.recording!=null&&(e.headers["X-webswing-recording"]=r.cfg.recording),r.cfg.debugPort!=null&&(e.headers["X-webswing-debugPort"]=r.cfg.debugPort),e.onOpen=function(e){e.transport!=="websocket"&&a&&(console.error("Webswing: Binary encoding not supported for "+e.transport+" transport. Falling back to json encoding."),r.cfg.binarySocket=!1,a=!1,h(),l())},e.onReopen=function(e){r.hideDialog()},e.onMessage=function(e){try{var t=c(e);t.sessionId!=null&&(u=t.sessionId);if(t.javaResponse!=null&&t.javaResponse.correlationId!=null){var n=t.javaResponse.correlationId;if(f[n]!=null){var i=f[n];delete f[n],i(t.javaResponse)}}r.processMessage(t)}catch(s){console.error(s);return}},e.onClose=function(e){r.currentDialog()!==r.stoppedDialog&&r.showDialog(r.disconnectedDialog)},e.onError=function(e){r.showDialog(r.connectionErrorDialog)},i=t.subscribe(e)}function c(e){var n=e.responseBody,r;if(a){if(n.byteLength===1)return{};r=o.decode(n),m(r)}else r=t.util.parseJSON(n);return r}function h(){t.unsubscribe(i),i=null,u=null}function p(e){if(i!=null&&i.request.isOpen&&!i.request.closed)if(typeof e=="object")if(a){var n=new s(e);i.push(n.encode().toArrayBuffer())}else i.push(t.util.stringifyJSON(e));else console.log("message is not an object "+e)}function d(e,t,n,r){p(t),f[n]=e,setTimeout(function(){f[n]!=null&&(delete f[n],e(new Error("Java call timed out after "+r+" ms.")))},r)}function v(){return u}function m(e){e!=null&&(Array.isArray(e)?e.forEach(function(e){m(e)}):e.$type._fields.forEach(function(t){if(t.resolvedType!=null)if(t.resolvedType.className==="Enum"){var n=t.resolvedType.object;for(var r in n)n[r]===e[t.name]&&(e[t.name]=r)}else t.resolvedType.className==="Message"&&m(e[t.name])}))}var n=this,r;n.injects=r={cfg:"webswing.config",processMessage:"base.processMessage",showDialog:"dialog.show",hideDialog:"dialog.hide",currentDialog:"dialog.current",stoppedDialog:"dialog.content.stoppedDialog",disconnectedDialog:"dialog.content.disconnectedDialog",connectionErrorDialog:"dialog.content.connectionErrorDialog",initializingDialog:"dialog.content.initializingDialog"},n.provides={connect:l,send:p,uuid:v,awaitResponse:d,dispose:h},n.ready=function(){a=r.cfg.typedArraysSupported&&r.cfg.binarySocket};var i,u,a,f={}}});