webswingRequirejs.define(["jquery","webswing-util"],function(t,n){return function(){function f(e){l();var t=a;a=[],e!=null&&t.push(e),t.length>0&&r.send({events:t})}function l(e){r.cfg.hasControl&&(i!=null&&(a.push(i),i=null),s!=null&&(a.push(s),s=null),o!=null&&(a.push(o),o=null),e!=null&&JSON.stringify(a[a.length-1])!==JSON.stringify(e)&&a.push(e))}function c(){i=null,s=null,o=null,u=0,a=[]}function h(){document.removeEventListener("mousedown",d),document.removeEventListener("mouseout",v),document.removeEventListener("mouseup",m)}function p(){var e=r.getCanvas(),t=r.getInput();c(),g(t),n.bindEvent(e,"mousedown",function(n){var r=y(e,n,"mousedown");return i=null,l(r),g(t),f(),!1},!1),n.bindEvent(e,"dblclick",function(n){var r=y(e,n,"dblclick");return i=null,l(r),g(t),f(),!1},!1),n.bindEvent(e,"mousemove",function(t){var n=y(e,t,"mousemove");return n.mouse.button=u,i=n,!1},!1),n.bindEvent(e,"mouseup",function(n){var r=y(e,n,"mouseup");return i=null,l(r),g(t),f(),!1},!1),n.bindEvent(e,"mousewheel",function(t){var n=y(e,t,"mousewheel");return i=null,s!=null&&(n.mouse.wheelDelta+=s.mouse.wheelDelta),s=n,!1},!1),n.bindEvent(e,"DOMMouseScroll",function(t){var n=y(e,t,"mousewheel");return i=null,s!=null&&(n.mouse.wheelDelta+=s.mouse.wheelDelta),s=n,!1},!1),n.bindEvent(e,"contextmenu",function(e){return e.preventDefault(),e.stopPropagation(),!1}),n.bindEvent(t,"keydown",function(t){var n=[9,12,16,17,18,19,20,27,32,33,34,35,36,37,38,39,40,44,45,46,91,92,93,145,225,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135],i=t.keyCode;n.indexOf(i)!=-1&&(r.cfg.virtualKB||(t.preventDefault(),t.stopPropagation()));var s=b("keydown",e,t);if(!s.key.ctrl||s.key.character!=88&&s.key.character!=67&&s.key.character!=86)s.key.ctrl&&!s.key.alt&&!s.key.altgr&&t.preventDefault(),l(s);return!1},!1),n.bindEvent(t,"keypress",function(t){var n=b("keypress",e,t);if(!n.key.ctrl||n.key.character!=120&&n.key.character!=24&&n.key.character!=99&&n.key.character!=118&&n.key.character!=22)t.preventDefault(),t.stopPropagation(),l(n);return!1},!1),n.bindEvent(t,"keyup",function(t){var n=b("keyup",e,t);if(!n.key.ctrl||n.key.character!=88&&n.key.character!=67&&n.key.character!=86)t.preventDefault(),t.stopPropagation(),l(n),f();return!1},!1),n.bindEvent(t,"cut",function(e){return e.preventDefault(),e.stopPropagation(),r.cut(e),!1},!1),n.bindEvent(t,"copy",function(e){return e.preventDefault(),e.stopPropagation(),r.copy(e),!1},!1),n.bindEvent(t,"paste",function(e){return e.preventDefault(),e.stopPropagation(),r.paste(e),!1},!1),n.bindEvent(document,"mousedown",d),n.bindEvent(document,"mouseout",v),n.bindEvent(document,"mouseup",m)}function d(e){e.which==1&&(u=1)}function v(e){u=0}function m(e){e.which==1&&(u=0)}function g(e){e.value=" ",e.focus(),e.select()}function y(e,t,n){var r=e.getBoundingClientRect(),i=Math.round(t.clientX-r.left),s=Math.round(t.clientY-r.top),o=0;return n=="mousewheel"&&(o=-Math.max(-1,Math.min(1,t.wheelDelta||-t.detail))),{mouse:{x:i,y:s,type:n,wheelDelta:o,button:t.which,ctrl:t.ctrlKey,alt:t.altKey,shift:t.shiftKey,meta:t.metaKey}}}function b(e,t,n){var r=n.which;r==0&&n.key!=null&&(r=n.key.charCodeAt(0));var i=n.keyCode;return i==0&&(i=r),{key:{type:e,character:r,keycode:i,alt:n.altKey,ctrl:n.ctrlKey,shift:n.shiftKey,meta:n.metaKey}}}var t=this,r;t.injects=r={cfg:"webswing.config",send:"socket.send",getInput:"canvas.getInput",getCanvas:"canvas.get",cut:"clipboard.cut",copy:"clipboard.copy",paste:"clipboard.paste"},t.provides={register:p,sendInput:f,dispose:h},t.ready=function(){};var i=null,s=null,o=null,u=0,a=[]}});