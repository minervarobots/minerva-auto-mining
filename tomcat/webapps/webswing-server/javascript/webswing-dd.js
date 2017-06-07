(function(e,t){typeof webswingRequirejs.define=="function"&&webswingRequirejs.define.amd?webswingRequirejs.define(["ProtoBuf","text!directdraw.proto"],t):e.WebswingDirectDraw=t(dcodeIO.ProtoBuf)})(this,function(e,t){return function(n){function g(e,t){return w(o.decode64(e),t)}function y(e,t){return w(o.decode(e),t)}function b(e,t){return w(e,t)}function w(e,t){return new Promise(function(n,i){try{E(e,t,n,i)}catch(s){r.onErrorMessage(s),i(s)}})}function E(e,t,n,i){var s;t!=null?s=t:s=document.createElement("canvas");if(s.width!=e.width||s.height!=e.height)s.width=e.width,s.height=e.height;var o={canvas:s,graphicsStates:{},currentStateId:null},u=S(e.constants);Z(u).then(function(){return et(e.fontFaces)}).then(function(){var t=o.canvas.getContext("2d");e.instructions!=null&&(t.save(),e.instructions.reduce(function(e,n){return e.then(function(e){return x(t,n,o)})},Promise.resolve()).then(function(){t.restore(),n(o.canvas)},function(e){t.restore(),i(e),r.onErrorMessage(e)}))},function(e){r.onErrorMessage(e)})}function S(e){var t=[];return e.forEach(function(e){v[e.id]=e,e.image!=null?t.push(e.image):e.texture!=null?t.push(e.texture.image):e.glyph!=null&&e.glyph.data!=null&&t.push(e.glyph)}),t}function x(e,t,n){var r=ot(t.args,v),i=n.graphicsStates[n.currentStateId];switch(t.inst){case u.GRAPHICS_CREATE:C(e,t.args[0],r,n);break;case u.GRAPHICS_SWITCH:N(e,t.args[0],n);break;case u.GRAPHICS_DISPOSE:T(t.args[0],n);break;case u.DRAW:k(e,r);break;case u.FILL:L(e,r);break;case u.DRAW_IMAGE:A(e,r);break;case u.DRAW_WEBIMAGE:return O(e,r,t.webImage);case u.DRAW_STRING:_(e,r,i.fontTransform);break;case u.COPY_AREA:P(e,r);break;case u.SET_STROKE:i.strokeArgs=r,B(e,r);break;case u.SET_PAINT:i.paintArgs=r,j(e,r);break;case u.SET_COMPOSITE:i.compositeArgs=r,z(e,r);break;case u.SET_FONT:i.fontArgs=r,i.fontTransform=D(e,r);break;case u.TRANSFORM:i.transform=ut(i.transform,H(e,r));break;case u.DRAW_GLYPH_LIST:M(e,r);break;default:console.log("instruction code: "+t.inst+" not recognized")}return Promise.resolve()}function T(e,t){delete t.graphicsStates[e]}function N(e,t,n){var r=n.graphicsStates;if(r[t]!=null){r[t].strokeArgs!=null&&B(e,r[t].strokeArgs),r[t].paintArgs!=null&&j(e,r[t].paintArgs),r[t].compositeArgs!=null&&z(e,r[t].compositeArgs),r[t].fontArgs!=null&&D(e,r[t].fontArgs);if(r[t].transform!=null){var i=r[t].transform;e.setTransform(i[0],i[1],i[2],i[3],i[4],i[5])}}else console.log("Graphics with id "+t+" not initialized!");n.currentStateId=t}function C(e,t,n,r){var i=r.graphicsStates;i[t]==null?(i[t]={},r.currentStateId=t,n.shift(),i[t].transform=H(e,n,!0),n.shift(),B(e,n),i[t].strokeArgs=n.slice(0,1),n.shift(),z(e,n),i[t].compositeArgs=n.slice(0,1),n.shift(),j(e,n),i[t].paintArgs=n.slice(0,1),n.shift(),i[t].fontArgs=n,i[t].fontTransform=D(e,n)):console.log("Graphics with id "+t+" already exist!")}function k(e,t){e.save(),W(e,t[1])&&e.clip(st(t[1])),W(e,t[0],!0),e.stroke(),e.restore()}function L(e,t){e.save(),W(e,t[1])&&e.clip(st(t[1])),W(e,t[0]),e.fill(st(t[0])),e.restore()}function A(e,t){e.save();var n=t[0].image.data,r=t[1],i=t[2],s=t[3],o=t[4];W(e,o)&&e.clip(st(o)),r!=null&&H(e,[r]),s!=null&&(e.fillStyle=Y(s.color.rgba),e.beginPath(),i==null?e.rect(0,0,n.width,n.height):e.rect(0,0,i.rectangle.w,i.rectangle.h),e.fill()),i==null?e.drawImage(n,0,0):(i=i.rectangle,e.drawImage(n,i.x,i.y,i.w,i.h,0,0,i.w,i.h)),e.restore()}function O(e,t,n){var r=t[0],i=t[1],s=t[2],o=t[3];return y(n).then(function(t){e.save(),W(e,o)&&e.clip(st(o)),r!=null&&H(e,[r]),s!=null&&(e.fillStyle=Y(s.color.rgba),e.beginPath(),i==null?e.rect(0,0,t.width,t.height):e.rect(0,0,i.rectangle.w,i.rectangle.h),e.fill()),i==null?e.drawImage(t,0,0):(i=i.rectangle,e.drawImage(t,i.x,i.y,i.w,i.h,0,0,i.w,i.h)),e.restore()})}function M(e,t){var n=ot(t[0].combined.ids,v),r=n[0].points,i=n[1].points,s=n.slice(2),o=t[1];e.save(),W(e,o)&&e.clip(st(o));if(s.length>0){var u=document.createElement("canvas");u.width=r.points[2],u.height=r.points[3];var a=u.getContext("2d");for(var f=0;f<s.length;f++)if(s[f].glyph.data!=null){var l=s[f].glyph.data,c=i.points[f*2],h=i.points[f*2+1];a.drawImage(l,0,0,l.width,l.height,c,h,l.width,l.height)}a.fillStyle=e.fillStyle,a.globalCompositeOperation="source-in",a.fillRect(0,0,u.width,u.height),e.drawImage(u,0,0,u.width,u.height,r.points[0],r.points[1],u.width,u.height)}e.restore()}function _(e,t,n){var r=t[0].string,i=t[1].points.points,s=t[2];e.save(),W(e,s)&&e.clip(st(s));if(n!=null){var o=n;e.transform(o.m00,o.m10,o.m01,o.m11,o.m02+i[0],o.m12+i[1]),e.fillText(r,0,0)}else{var u=e.measureText(r).width,a=i[2]/u;e.scale(a,1),e.fillText(r,i[0]/a,i[1])}e.restore()}function D(e,t){if(t[0]==null)return e.font;var n=t[0].font,r="";switch(n.style){case p.NORMAL:r="";break;case p.OBLIQUE:r="bold";break;case p.ITALIC:r="italic";break;case p.BOLDANDITALIC:r="bold italic"}var s=n.family;return n.family!=="sans-serif"&&n.family!=="serif"&&n.family!=="monospace"&&(s='"'+i+n.family+'"'),e.font=r+" "+n.size+"px "+s,n.transform}function P(e,t){var n=t[0].points.points,r=t[1];e.save(),W(e,r)&&e.clip(st(r)),e.beginPath(),e.setTransform(1,0,0,1,0,0),e.rect(n[0],n[1],n[2],n[3]),e.clip(),e.translate(n[4],n[5]),e.drawImage(e.canvas,0,0),e.restore()}function H(e,t,n){var r=t[0].transform;return n?e.setTransform(r.m00,r.m10,r.m01,r.m11,r.m02,r.m12):e.transform(r.m00,r.m10,r.m01,r.m11,r.m02,r.m12),[r.m00,r.m10,r.m01,r.m11,r.m02,r.m12]}function B(e,t){var n=t[0].stroke;e.lineWidth=n.width,e.miterLimit=n.miterLimit;switch(n.cap){case h.CAP_BUTT:e.lineCap="butt";break;case h.CAP_ROUND:e.lineCap="round";break;case h.CAP_SQUARE:e.lineCap="square"}switch(n.join){case c.JOIN_MITER:e.lineJoin="miter";break;case c.JOIN_ROUND:e.lineJoin="round";break;case c.JOIN_BEVEL:e.lineJoin="bevel"}n.dash!=null&&(e.setLineDash(n.dash),e.lineDashOffset=n.dashOffset)}function j(e,t){var n=t[0];if(n.color!=null){var r=Y(n.color.rgba);e.fillStyle=r,e.strokeStyle=r}else if(n.texture!=null){var i=n.texture.anchor,s=n.texture.image.data,o;if(i.x==0&&i.y==0&&i.w==s.width&&i==s.height)o=e.createPattern(s,"repeat");else{var u=document.createElement("canvas"),a=i.x<0?i.x%i.w+i.w:i.x%i.w,f=i.y<0?i.y%i.h+i.h:i.y%i.h;u.width=i.w,u.height=i.h;var l=u.getContext("2d");l.fillRect(0,0,i.w,i.h),l.fillStyle=l.createPattern(s,"repeat"),l.setTransform(i.w/s.width,0,0,i.h/s.height,a,f),l.fillRect(-a*s.width/i.w,-f*s.height/i.h,s.width,s.height),o=e.createPattern(u,"repeat")}e.fillStyle=o,e.strokeStyle=o}else if(n.linearGrad!=null){var c=F(e,n.linearGrad);e.fillStyle=c,e.strokeStyle=c}else if(n.radialGrad!=null){var c=q(e,n.radialGrad);e.fillStyle=c,e.strokeStyle=c}}function F(e,t){var n=t.xStart,r=t.yStart,i=t.xEnd-n,s=t.yEnd-r,o=1,u=o,a=0;if(t.repeat!=l.NO_CYCLE&&(i!=0||s!=0)){var f=e.canvas,c=[I(n,r,i,s,0,0),I(n,r,i,s,f.width,0),I(n,r,i,s,f.width,f.height),I(n,r,i,s,0,f.height)];u=Math.ceil(Math.max.apply(Math,c)),a=Math.ceil(-Math.min.apply(Math,c)),o=u+a}var h=e.createLinearGradient(n-i*a,r-s*a,n+i*u,r+s*u);for(var p=-a,d=0;p<u;p++,d++)if(t.repeat!=l.REFLECT||p%2==0)for(var v=0;v<t.colors.length;v++)h.addColorStop((d+t.fractions[v])/o,Y(t.colors[v]));else for(var v=t.colors.length-1;v>=0;v--)h.addColorStop((d+(1-t.fractions[v]))/o,Y(t.colors[v]));return h}function I(e,t,n,r,i,s){return((i-e)*n+(s-t)*r)/(n*n+r*r)}function q(e,t){R(t);var n=t.xFocus,r=t.yFocus,i=t.xCenter-n,s=t.yCenter-r,o=t.radius,u=1;if(t.repeat!=l.NO_CYCLE)if(i==0&&s==0){var a=e.canvas,f=[U(n,r,0,0)/o,U(n,r,a.width,0)/o,U(n,r,a.width,a.height)/o,U(n,r,0,a.height)/o];u=Math.ceil(Math.max.apply(Math,f))}else{var c=Math.sqrt(i*i+s*s),h=i+o*i/c,p=s+o*s/c,d=i-o*i/c,v=s-o*s/c,a=e.canvas,f=[I(n,r,h,p,0,0),I(n,r,h,p,a.width,0),I(n,r,h,p,a.width,a.height),I(n,r,h,p,0,a.height),I(n,r,d,v,0,0),I(n,r,d,v,a.width,0),I(n,r,d,v,a.width,a.height),I(n,r,d,v,0,a.height)];u=Math.ceil(Math.max.apply(Math,f))}var m=e.createRadialGradient(n,r,0,n+u*i,r+u*s,o*u);for(var g=0;g<u;g++)if(t.repeat!=l.REFLECT||g%2==0)for(var y=0;y<t.colors.length;y++)m.addColorStop((g+t.fractions[y])/u,Y(t.colors[y]));else for(var y=t.colors.length-1;y>=0;y--)m.addColorStop((g+(1-t.fractions[y]))/u,Y(t.colors[y]));return m}function R(e){var t=e.xFocus-e.xCenter,n=e.yFocus-e.yCenter;if(t==0&&n==0)return;var r=.99,i=e.radius*e.radius,s=t*t+n*n;if(s>i*r){var o=Math.sqrt(i*r/s);e.xFocus=e.xCenter+t*o,e.yFocus=e.yCenter+n*o}}function U(e,t,n,r){return Math.sqrt((n-e)*(n-e)+(r-t)*(r-t))}function z(e,t){var n=t[0].composite;if(n!=null){e.globalAlpha=n.alpha;switch(n.type){case d.CLEAR:e.globalCompositeOperation="destination-out",e.globalAlpha=1;break;case d.SRC:e.globalCompositeOperation="source-over";break;case d.DST:e.globalCompositeOperation="destination-over",e.globalAlpha=0;break;case d.SRC_OVER:e.globalCompositeOperation="source-over";break;case d.DST_OVER:e.globalCompositeOperation="destination-over";break;case d.SRC_IN:e.globalCompositeOperation="source-in";break;case d.DST_IN:e.globalCompositeOperation="destination-in";break;case d.SRC_OUT:e.globalCompositeOperation="source-out";break;case d.DST_OUT:e.globalCompositeOperation="destination-out";break;case d.SRC_ATOP:e.globalCompositeOperation="source-atop";break;case d.DST_ATOP:e.globalCompositeOperation="destination-atop";break;case d.XOR:e.globalCompositeOperation="xor"}}}function W(e,t,n){if(t==null)return!1;if(t.rectangle!=null)return e.beginPath(),X(e,t.rectangle,n),!0;if(t.roundRectangle!=null)return e.beginPath(),J(e,t.roundRectangle,n),!0;if(t.ellipse!=null)return e.beginPath(),V(e,t.ellipse,n),!0;if(t.arc!=null)return e.beginPath(),Q(e,t.arc,n),!0;if(t.path!=null){e.beginPath();var r=t.path,i=G(e,n),s=0;return r.type.forEach(function(t,n){switch(t){case a.MOVE:e.moveTo(r.points[s+0]+i,r.points[s+1]+i),s+=2;break;case a.LINE:e.lineTo(r.points[s+0]+i,r.points[s+1]+i),s+=2;break;case a.QUAD:e.quadraticCurveTo(r.points[s+0]+i,r.points[s+1]+i,r.points[s+2]+i,r.points[s+3]+i),s+=4;break;case a.CUBIC:e.bezierCurveTo(r.points[s+0]+i,r.points[s+1]+i,r.points[s+2]+i,r.points[s+3]+i,r.points[s+4]+i,r.points[s+5]+i),s+=6;break;case a.CLOSE:e.closePath();break;default:console.log("segment.type:"+segment.type+" not recognized")}}),!0}return!1}function X(e,t,n){var r=G(e,n);e.rect(t.x+r,t.y+r,t.w,t.h)}function V(e,t,n){var r=G(e,n),i=.5522847498307933,s=.5+i*.5,o=.5-i*.5;e.moveTo(t.x+r+t.w,t.y+r+.5*t.h);var u=$([1,s,s,1,.5,1],t,r);e.bezierCurveTo(u[0],u[1],u[2],u[3],u[4],u[5]),u=$([o,1,0,s,0,.5],t,r),e.bezierCurveTo(u[0],u[1],u[2],u[3],u[4],u[5]),u=$([0,o,o,0,.5,0],t,r),e.bezierCurveTo(u[0],u[1],u[2],u[3],u[4],u[5]),u=$([s,0,1,o,1,.5],t,r),e.bezierCurveTo(u[0],u[1],u[2],u[3],u[4],u[5]),e.closePath()}function $(e,t,n){return e[0]=t.x+n+e[0]*t.w,e[1]=t.y+n+e[1]*t.h,e[2]=t.x+n+e[2]*t.w,e[3]=t.y+n+e[3]*t.h,e[4]=t.x+n+e[4]*t.w,e[5]=t.y+n+e[5]*t.h,e}function J(e,t,n){var r=G(e,n),i=.22385762508460333,s=K([0,0,0,.5],t,r);e.moveTo(s[0],s[1]),s=K([0,0,1,-0.5],t,r),e.lineTo(s[0],s[1]),s=K([0,0,1,-i,0,i,1,0,0,.5,1,0],t,r),e.bezierCurveTo(s[0],s[1],s[2],s[3],s[4],s[5]),s=K([1,-0.5,1,0],t,r),e.lineTo(s[0],s[1]),s=K([1,-i,1,0,1,0,1,-i,1,0,1,-0.5],t,r),e.bezierCurveTo(s[0],s[1],s[2],s[3],s[4],s[5]),s=K([1,0,0,.5],t,r),e.lineTo(s[0],s[1]),s=K([1,0,0,i,1,-i,0,0,1,-0.5,0,0],t,r),e.bezierCurveTo(s[0],s[1],s[2],s[3],s[4],s[5]),s=K([0,.5,0,0],t,r),e.lineTo(s[0],s[1]),s=K([0,i,0,0,0,0,0,i,0,0,0,.5],t,r),e.bezierCurveTo(s[0],s[1],s[2],s[3],s[4],s[5]),e.closePath()}function K(e,t,n){var r=[],i=0;for(var s=0;s<e.length;s+=4)r[i++]=t.x+n+e[s+0]*t.w+e[s+1]*Math.abs(t.arcW),r[i++]=t.y+n+e[s+2]*t.h+e[s+3]*Math.abs(t.arcH);return r}function Q(e,t,n){var r=G(e,n),i=t.w/2,s=t.h/2,o=t.x+r+i,u=t.y+r+s,a=-(t.start*Math.PI/180),l=-t.extent,c=4,h=l<0?Math.PI/2:-Math.PI/2,p=l<0?.5522847498307933:-0.5522847498307933;l>-360&&l<360&&(c=Math.ceil(Math.abs(l)/90),h=l/c*Math.PI/180,p=4/3*Math.sin(h/2)/(1+Math.cos(h/2)),c=p==0?0:c),e.moveTo(o+Math.cos(a)*i,u+Math.sin(a)*s);for(var d=0;d<c;d++){var v=a+h*d,m=Math.cos(v),g=Math.sin(v),y=[];y[0]=o+(m-p*g)*i,y[1]=u+(g+p*m)*s,v+=h,m=Math.cos(v),g=Math.sin(v),y[2]=o+(m+p*g)*i,y[3]=u+(g-p*m)*s,y[4]=o+m*i,y[5]=u+g*s,e.bezierCurveTo(y[0],y[1],y[2],y[3],y[4],y[5])}switch(t.type){case f.OPEN:break;case f.CHORD:e.closePath();break;case f.PIE:e.lineTo(o,u),e.closePath()}}function G(e,t){return e.lineWidth&1&&t?.5:0}function Y(e){var t=255;return"rgba("+(e>>>24&t)+","+(e>>>16&t)+","+(e>>>8&t)+","+(e&t)/255+")"}function Z(e){return new Promise(function(t,n){try{nt(e,t,n)}catch(i){r.onErrorMessage(error),n(i)}})}function et(e){return new Promise(function(t,n){try{if(e.length>0){var s=e.map(function(e){return new Promise(function(t){if(m.indexOf(e.name)>=0)t();else{m.push(e.name);var n=document.createElement("style");n.type="text/css",n.setAttribute("data-dd-ctx",i),n.innerHTML=tt(i+e.name,e.font,e.style),document.body.appendChild(n),t()}})});Promise.all(s).then(t)}else t()}catch(o){r.onErrorMessage(error),n(o)}})}function tt(e,t,n){var r="@font-face {";return r+="font-family: '"+e+"';",r+="src: url(data:font/truetype;base64,"+rt(t)+");",n!=null&&(r+="font-style: "+n+";"),r+="}",r}function nt(e,t,n){if(e.length>0){var r=e.map(function(e){return new Promise(function(t){var n=new Image;n.onload=function(){e.data=n,t()},n.src=it(e)})});Promise.all(r).then(t)}else t()}function rt(e){var t="",n=new Uint8Array(e.buffer,e.offset,e.limit-e.offset);for(var r=0,i=n.byteLength;r<i;r++)t+=String.fromCharCode(n[r]);return window.btoa(t)}function it(e){return"data:image/png;base64,"+rt(e.data)}function st(e){return e.path!=null?e.path.windingOdd?"evenodd":"nonzero":"nonzero"}function ot(e,t){var n=[];for(var r=0;r<e.length;r++)n[r]=v[e[r]];return n}function ut(e,t){var n=[];return e==null?t:(n[0]=e[0]*t[0]+e[2]*t[1],n[1]=e[1]*t[0]+e[3]*t[1],n[2]=e[0]*t[2]+e[2]*t[3],n[3]=e[1]*t[2]+e[3]*t[3],n[4]=e[0]*t[4]+e[2]*t[5]+e[4],n[5]=e[1]*t[4]+e[3]*t[5]+e[5],n)}function at(){var e=document.body.getElementsByTagName("style"),t=[];for(var n=0;n<e.length;n++)e[n].getAttribute("data-dd-ctx")===i&&t.push(e[n]);t.forEach(function(e){document.body.removeChild(e)})}n=n||{};var r={onErrorMessage:n.onErrorMessage||function(e){console.log(e.stack)}},i=Math.floor(Math.random()*65536).toString(16),s=t!=null?e.loadProto(t,"directdraw.proto"):e.loadProtoFile("/directdraw.proto"),o=s.build("org.webswing.directdraw.proto.WebImageProto"),u=s.build("org.webswing.directdraw.proto.DrawInstructionProto.InstructionProto"),a=s.build("org.webswing.directdraw.proto.PathProto.SegmentTypeProto"),f=s.build("org.webswing.directdraw.proto.ArcProto.ArcTypeProto"),l=s.build("org.webswing.directdraw.proto.CyclicMethodProto"),c=s.build("org.webswing.directdraw.proto.StrokeProto.StrokeJoinProto"),h=s.build("org.webswing.directdraw.proto.StrokeProto.StrokeCapProto"),p=s.build("org.webswing.directdraw.proto.FontProto.StyleProto"),d=s.build("org.webswing.directdraw.proto.CompositeProto.CompositeTypeProto"),v=n.constantPoolCache||{},m=[];return{draw64:g,drawBin:y,drawProto:b,dispose:at,getConstantPoolCache:function(){return v}}}});