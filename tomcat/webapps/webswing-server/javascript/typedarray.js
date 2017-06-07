/*
 Copyright (c) 2010, Linden Research, Inc.
 Copyright (c) 2014, Joshua Bell

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
 $/LicenseInfo$
 */

(function(e){function r(e){switch(typeof e){case"undefined":return"undefined";case"boolean":return"boolean";case"number":return"number";case"string":return"string";default:return e===null?"null":"object"}}function i(e){return Object.prototype.toString.call(e).replace(/^\[object *|\]$/g,"")}function s(e){return typeof e=="function"}function o(e){if(e===null||e===t)throw TypeError();return Object(e)}function u(e){return e>>0}function a(e){return e>>>0}function g(e){function t(t){Object.defineProperty(e,t,{get:function(){return e._getter(t)},set:function(n){e._setter(t,n)},enumerable:!0,configurable:!1})}if(e.length>n)throw RangeError("Array too large for polyfill");var r;for(r=0;r<e.length;r+=1)t(r)}function y(e,t){var n=32-t;return e<<n>>n}function b(e,t){var n=32-t;return e<<n>>>n}function w(e){return[e&255]}function E(e){return y(e[0],8)}function S(e){return[e&255]}function x(e){return b(e[0],8)}function T(e){return e=m(Number(e)),[e<0?0:e>255?255:e&255]}function N(e){return[e>>8&255,e&255]}function C(e){return y(e[0]<<8|e[1],16)}function k(e){return[e>>8&255,e&255]}function L(e){return b(e[0]<<8|e[1],16)}function A(e){return[e>>24&255,e>>16&255,e>>8&255,e&255]}function O(e){return y(e[0]<<24|e[1]<<16|e[2]<<8|e[3],32)}function M(e){return[e>>24&255,e>>16&255,e>>8&255,e&255]}function _(e){return b(e[0]<<24|e[1]<<16|e[2]<<8|e[3],32)}function D(e,t,n){function y(e){var t=c(e),n=e-t;return n<.5?t:n>.5?t+1:t%2?t+1:t}var r=(1<<t-1)-1,i,s,o,u,a,p,m,g;e!==e?(s=(1<<t)-1,o=v(2,n-1),i=0):e===Infinity||e===-Infinity?(s=(1<<t)-1,o=0,i=e<0?1:0):e===0?(s=0,o=0,i=1/e===-Infinity?1:0):(i=e<0,e=l(e),e>=v(2,1-r)?(s=d(c(h(e)/f),1023),o=y(e/v(2,s)*v(2,n)),o/v(2,n)>=2&&(s+=1,o=1),s>r?(s=(1<<t)-1,o=0):(s+=r,o-=v(2,n))):(s=0,o=y(e/v(2,1-r-n)))),p=[];for(a=n;a;a-=1)p.push(o%2?1:0),o=c(o/2);for(a=t;a;a-=1)p.push(s%2?1:0),s=c(s/2);p.push(i?1:0),p.reverse(),m=p.join(""),g=[];while(m.length)g.push(parseInt(m.substring(0,8),2)),m=m.substring(8);return g}function P(e,t,n){var r=[],i,s,o,u,a,f,l,c;for(i=e.length;i;i-=1){o=e[i-1];for(s=8;s;s-=1)r.push(o%2?1:0),o>>=1}return r.reverse(),u=r.join(""),a=(1<<t-1)-1,f=parseInt(u.substring(0,1),2)?-1:1,l=parseInt(u.substring(1,1+t),2),c=parseInt(u.substring(1+t),2),l===(1<<t)-1?c!==0?NaN:f*Infinity:l>0?f*v(2,l-a)*(1+c/v(2,n)):c!==0?f*v(2,-(a-1))*(c/v(2,n)):f<0?0:0}function H(e){return P(e,11,52)}function B(e){return D(e,11,52)}function j(e){return P(e,8,23)}function F(e){return D(e,8,23)}var t=void 0,n=1e5,f=Math.LN2,l=Math.abs,c=Math.floor,h=Math.log,p=Math.max,d=Math.min,v=Math.pow,m=Math.round;(function(){var e=Object.defineProperty,t=!function(){try{return Object.defineProperty({},"x",{})}catch(e){return!1}}();if(!e||t)Object.defineProperty=function(t,n,r){if(e)try{return e(t,n,r)}catch(i){}if(t!==Object(t))throw TypeError("Object.defineProperty called on non-object");return Object.prototype.__defineGetter__&&"get"in r&&Object.prototype.__defineGetter__.call(t,n,r.get),Object.prototype.__defineSetter__&&"set"in r&&Object.prototype.__defineSetter__.call(t,n,r.set),"value"in r&&(t[n]=r.value),t}})(),function(){function n(e){e=u(e);if(e<0)throw RangeError("ArrayBuffer size is not a small enough positive integer.");Object.defineProperty(this,"byteLength",{value:e}),Object.defineProperty(this,"_bytes",{value:Array(e)});for(var t=0;t<e;t+=1)this._bytes[t]=0}function f(){if(!arguments.length||typeof arguments[0]!="object")return function(e){e=u(e);if(e<0)throw RangeError("length is not a small enough positive integer.");Object.defineProperty(this,"length",{value:e}),Object.defineProperty(this,"byteLength",{value:e*this.BYTES_PER_ELEMENT}),Object.defineProperty(this,"buffer",{value:new n(this.byteLength)}),Object.defineProperty(this,"byteOffset",{value:0})}.apply(this,arguments);if(arguments.length>=1&&r(arguments[0])==="object"&&arguments[0]instanceof f)return function(e){if(this.constructor!==e.constructor)throw TypeError();var t=e.length*this.BYTES_PER_ELEMENT;Object.defineProperty(this,"buffer",{value:new n(t)}),Object.defineProperty(this,"byteLength",{value:t}),Object.defineProperty(this,"byteOffset",{value:0}),Object.defineProperty(this,"length",{value:e.length});for(var r=0;r<this.length;r+=1)this._setter(r,e._getter(r))}.apply(this,arguments);if(!(arguments.length>=1&&r(arguments[0])==="object")||arguments[0]instanceof f||arguments[0]instanceof n||i(arguments[0])==="ArrayBuffer"){if(arguments.length>=1&&r(arguments[0])==="object"&&(arguments[0]instanceof n||i(arguments[0])==="ArrayBuffer"))return function(e,n,r){n=a(n);if(n>e.byteLength)throw RangeError("byteOffset out of range");if(n%this.BYTES_PER_ELEMENT)throw RangeError("buffer length minus the byteOffset is not a multiple of the element size.");if(r===t){var i=e.byteLength-n;if(i%this.BYTES_PER_ELEMENT)throw RangeError("length of buffer minus byteOffset not a multiple of the element size");r=i/this.BYTES_PER_ELEMENT}else r=a(r),i=r*this.BYTES_PER_ELEMENT;if(n+i>e.byteLength)throw RangeError("byteOffset and length reference an area beyond the end of the buffer");Object.defineProperty(this,"buffer",{value:e}),Object.defineProperty(this,"byteLength",{value:i}),Object.defineProperty(this,"byteOffset",{value:n}),Object.defineProperty(this,"length",{value:r})}.apply(this,arguments);throw TypeError()}return function(e){var t=e.length*this.BYTES_PER_ELEMENT;Object.defineProperty(this,"buffer",{value:new n(t)}),Object.defineProperty(this,"byteLength",{value:t}),Object.defineProperty(this,"byteOffset",{value:0}),Object.defineProperty(this,"length",{value:e.length});for(var r=0;r<this.length;r+=1){var i=e[r];this._setter(r,Number(i))}}.apply(this,arguments)}function v(e,t,n){var r=function(){Object.defineProperty(this,"constructor",{value:r}),f.apply(this,arguments),g(this)};"__proto__"in r?r.__proto__=f:(r.from=f.from,r.of=f.of),r.BYTES_PER_ELEMENT=e;var i=function(){};return i.prototype=h,r.prototype=new i,Object.defineProperty(r.prototype,"BYTES_PER_ELEMENT",{value:e}),Object.defineProperty(r.prototype,"_pack",{value:t}),Object.defineProperty(r.prototype,"_unpack",{value:n}),r}e.ArrayBuffer=e.ArrayBuffer||n,Object.defineProperty(f,"from",{value:function(e){return new this(e)}}),Object.defineProperty(f,"of",{value:function(){return new this(arguments)}});var h={};f.prototype=h,Object.defineProperty(f.prototype,"_getter",{value:function(e){if(arguments.length<1)throw SyntaxError("Not enough arguments");e=a(e);if(e>=this.length)return t;var n=[],r,i;for(r=0,i=this.byteOffset+e*this.BYTES_PER_ELEMENT;r<this.BYTES_PER_ELEMENT;r+=1,i+=1)n.push(this.buffer._bytes[i]);return this._unpack(n)}}),Object.defineProperty(f.prototype,"get",{value:f.prototype._getter}),Object.defineProperty(f.prototype,"_setter",{value:function(e,t){if(arguments.length<2)throw SyntaxError("Not enough arguments");e=a(e);if(e>=this.length)return;var n=this._pack(t),r,i;for(r=0,i=this.byteOffset+e*this.BYTES_PER_ELEMENT;r<this.BYTES_PER_ELEMENT;r+=1,i+=1)this.buffer._bytes[i]=n[r]}}),Object.defineProperty(f.prototype,"constructor",{value:f}),Object.defineProperty(f.prototype,"copyWithin",{value:function(e,n){var r=arguments[2],i=o(this),s=i.length,f=a(s);f=p(f,0);var l=u(e),c;l<0?c=p(f+l,0):c=d(l,f);var h=u(n),v;h<0?v=p(f+h,0):v=d(h,f);var m;r===t?m=f:m=u(r);var g;m<0?g=p(f+m,0):g=d(m,f);var y=d(g-v,f-c),b;v<c&&c<v+y?(b=-1,v=v+y-1,c=c+y-1):b=1;while(y>0)i._setter(c,i._getter(v)),v+=b,c+=b,y-=1;return i}}),Object.defineProperty(f.prototype,"every",{value:function(e){if(this===t||this===null)throw TypeError();var n=Object(this),r=a(n.length);if(!s(e))throw TypeError();var i=arguments[1];for(var o=0;o<r;o++)if(!e.call(i,n._getter(o),o,n))return!1;return!0}}),Object.defineProperty(f.prototype,"fill",{value:function(e){var n=arguments[1],r=arguments[2],i=o(this),s=i.length,f=a(s);f=p(f,0);var l=u(n),c;l<0?c=p(f+l,0):c=d(l,f);var h;r===t?h=f:h=u(r);var v;h<0?v=p(f+h,0):v=d(h,f);while(c<v)i._setter(c,e),c+=1;return i}}),Object.defineProperty(f.prototype,"filter",{value:function(e){if(this===t||this===null)throw TypeError();var n=Object(this),r=a(n.length);if(!s(e))throw TypeError();var i=[],o=arguments[1];for(var u=0;u<r;u++){var f=n._getter(u);e.call(o,f,u,n)&&i.push(f)}return new this.constructor(i)}}),Object.defineProperty(f.prototype,"find",{value:function(e){var n=o(this),r=n.length,i=a(r);if(!s(e))throw TypeError();var u=arguments.length>1?arguments[1]:t,f=0;while(f<i){var l=n._getter(f),c=e.call(u,l,f,n);if(Boolean(c))return l;++f}return t}}),Object.defineProperty(f.prototype,"findIndex",{value:function(e){var n=o(this),r=n.length,i=a(r);if(!s(e))throw TypeError();var u=arguments.length>1?arguments[1]:t,f=0;while(f<i){var l=n._getter(f),c=e.call(u,l,f,n);if(Boolean(c))return f;++f}return-1}}),Object.defineProperty(f.prototype,"forEach",{value:function(e){if(this===t||this===null)throw TypeError();var n=Object(this),r=a(n.length);if(!s(e))throw TypeError();var i=arguments[1];for(var o=0;o<r;o++)e.call(i,n._getter(o),o,n)}}),Object.defineProperty(f.prototype,"indexOf",{value:function(e){if(this===t||this===null)throw TypeError();var n=Object(this),r=a(n.length);if(r===0)return-1;var i=0;arguments.length>0&&(i=Number(arguments[1]),i!==i?i=0:i!==0&&i!==1/0&&i!==-1/0&&(i=(i>0||-1)*c(l(i))));if(i>=r)return-1;var s=i>=0?i:p(r-l(i),0);for(;s<r;s++)if(n._getter(s)===e)return s;return-1}}),Object.defineProperty(f.prototype,"join",{value:function(e){if(this===t||this===null)throw TypeError();var n=Object(this),r=a(n.length),i=Array(r);for(var s=0;s<r;++s)i[s]=n._getter(s);return i.join(e===t?",":e)}}),Object.defineProperty(f.prototype,"lastIndexOf",{value:function(e){if(this===t||this===null)throw TypeError();var n=Object(this),r=a(n.length);if(r===0)return-1;var i=r;arguments.length>1&&(i=Number(arguments[1]),i!==i?i=0:i!==0&&i!==1/0&&i!==-1/0&&(i=(i>0||-1)*c(l(i))));var s=i>=0?d(i,r-1):r-l(i);for(;s>=0;s--)if(n._getter(s)===e)return s;return-1}}),Object.defineProperty(f.prototype,"map",{value:function(e){if(this===t||this===null)throw TypeError();var n=Object(this),r=a(n.length);if(!s(e))throw TypeError();var i=[];i.length=r;var o=arguments[1];for(var u=0;u<r;u++)i[u]=e.call(o,n._getter(u),u,n);return new this.constructor(i)}}),Object.defineProperty(f.prototype,"reduce",{value:function(e){if(this===t||this===null)throw TypeError();var n=Object(this),r=a(n.length);if(!s(e))throw TypeError();if(r===0&&arguments.length===1)throw TypeError();var i=0,o;arguments.length>=2?o=arguments[1]:o=n._getter(i++);while(i<r)o=e.call(t,o,n._getter(i),i,n),i++;return o}}),Object.defineProperty(f.prototype,"reduceRight",{value:function(e){if(this===t||this===null)throw TypeError();var n=Object(this),r=a(n.length);if(!s(e))throw TypeError();if(r===0&&arguments.length===1)throw TypeError();var i=r-1,o;arguments.length>=2?o=arguments[1]:o=n._getter(i--);while(i>=0)o=e.call(t,o,n._getter(i),i,n),i--;return o}}),Object.defineProperty(f.prototype,"reverse",{value:function(){if(this===t||this===null)throw TypeError();var e=Object(this),n=a(e.length),r=c(n/2);for(var i=0,s=n-1;i<r;++i,--s){var o=e._getter(i);e._setter(i,e._getter(s)),e._setter(s,o)}return e}}),Object.defineProperty(f.prototype,"set",{value:function(e,t){if(arguments.length<1)throw SyntaxError("Not enough arguments");var n,r,i,s,o,u,f,l,c,h;if(typeof arguments[0]=="object"&&arguments[0].constructor===this.constructor){n=arguments[0],i=a(arguments[1]);if(i+n.length>this.length)throw RangeError("Offset plus length of array is out of range");l=this.byteOffset+i*this.BYTES_PER_ELEMENT,c=n.length*this.BYTES_PER_ELEMENT;if(n.buffer===this.buffer){h=[];for(o=0,u=n.byteOffset;o<c;o+=1,u+=1)h[o]=n.buffer._bytes[u];for(o=0,f=l;o<c;o+=1,f+=1)this.buffer._bytes[f]=h[o]}else for(o=0,u=n.byteOffset,f=l;o<c;o+=1,u+=1,f+=1)this.buffer._bytes[f]=n.buffer._bytes[u]}else{if(typeof arguments[0]!="object"||typeof arguments[0].length=="undefined")throw TypeError("Unexpected argument type(s)");r=arguments[0],s=a(r.length),i=a(arguments[1]);if(i+s>this.length)throw RangeError("Offset plus length of array is out of range");for(o=0;o<s;o+=1)u=r[o],this._setter(i+o,Number(u))}}}),Object.defineProperty(f.prototype,"slice",{value:function(e,n){var r=o(this),i=r.length,s=a(i),f=u(e),l=f<0?p(s+f,0):d(f,s),c=n===t?s:u(n),h=c<0?p(s+c,0):d(c,s),v=h-l,m=r.constructor,g=new m(v),y=0;while(l<h){var b=r._getter(l);g._setter(y,b),++l,++y}return g}}),Object.defineProperty(f.prototype,"some",{value:function(e){if(this===t||this===null)throw TypeError();var n=Object(this),r=a(n.length);if(!s(e))throw TypeError();var i=arguments[1];for(var o=0;o<r;o++)if(e.call(i,n._getter(o),o,n))return!0;return!1}}),Object.defineProperty(f.prototype,"sort",{value:function(e){if(this===t||this===null)throw TypeError();var n=Object(this),r=a(n.length),i=Array(r);for(var s=0;s<r;++s)i[s]=n._getter(s);e?i.sort(e):i.sort();for(s=0;s<r;++s)n._setter(s,i[s]);return n}}),Object.defineProperty(f.prototype,"subarray",{value:function(e,t){function n(e,t,n){return e<t?t:e>n?n:e}e=u(e),t=u(t),arguments.length<1&&(e=0),arguments.length<2&&(t=this.length),e<0&&(e=this.length+e),t<0&&(t=this.length+t),e=n(e,0,this.length),t=n(t,0,this.length);var r=t-e;return r<0&&(r=0),new this.constructor(this.buffer,this.byteOffset+e*this.BYTES_PER_ELEMENT,r)}});var m=v(1,w,E),y=v(1,S,x),b=v(1,T,x),D=v(2,N,C),P=v(2,k,L),I=v(4,A,O),q=v(4,M,_),R=v(4,F,j),U=v(8,B,H);e.Int8Array=e.Int8Array||m,e.Uint8Array=e.Uint8Array||y,e.Uint8ClampedArray=e.Uint8ClampedArray||b,e.Int16Array=e.Int16Array||D,e.Uint16Array=e.Uint16Array||P,e.Int32Array=e.Int32Array||I,e.Uint32Array=e.Uint32Array||q,e.Float32Array=e.Float32Array||R,e.Float64Array=e.Float64Array||U}(),function(){function n(e,t){return s(e.get)?e.get(t):e[t]}function o(e,n,r){if(!(e instanceof ArrayBuffer||i(e)==="ArrayBuffer"))throw TypeError();n=a(n);if(n>e.byteLength)throw RangeError("byteOffset out of range");r===t?r=e.byteLength-n:r=a(r);if(n+r>e.byteLength)throw RangeError("byteOffset and length reference an area beyond the end of the buffer");Object.defineProperty(this,"buffer",{value:e}),Object.defineProperty(this,"byteLength",{value:r}),Object.defineProperty(this,"byteOffset",{value:n})}function u(e){return function(i,s){i=a(i);if(i+e.BYTES_PER_ELEMENT>this.byteLength)throw RangeError("Array index out of range");i+=this.byteOffset;var o=new Uint8Array(this.buffer,i,e.BYTES_PER_ELEMENT),u=[];for(var f=0;f<e.BYTES_PER_ELEMENT;f+=1)u.push(n(o,f));return Boolean(s)===Boolean(r)&&u.reverse(),n(new e((new Uint8Array(u)).buffer),0)}}function f(e){return function(i,s,o){i=a(i);if(i+e.BYTES_PER_ELEMENT>this.byteLength)throw RangeError("Array index out of range");var u=new e([s]),f=new Uint8Array(u.buffer),l=[],c,h;for(c=0;c<e.BYTES_PER_ELEMENT;c+=1)l.push(n(f,c));Boolean(o)===Boolean(r)&&l.reverse(),h=new Uint8Array(this.buffer,i,e.BYTES_PER_ELEMENT),h.set(l)}}var r=function(){var e=new Uint16Array([4660]),t=new Uint8Array(e.buffer);return n(t,0)===18}();Object.defineProperty(o.prototype,"getUint8",{value:u(Uint8Array)}),Object.defineProperty(o.prototype,"getInt8",{value:u(Int8Array)}),Object.defineProperty(o.prototype,"getUint16",{value:u(Uint16Array)}),Object.defineProperty(o.prototype,"getInt16",{value:u(Int16Array)}),Object.defineProperty(o.prototype,"getUint32",{value:u(Uint32Array)}),Object.defineProperty(o.prototype,"getInt32",{value:u(Int32Array)}),Object.defineProperty(o.prototype,"getFloat32",{value:u(Float32Array)}),Object.defineProperty(o.prototype,"getFloat64",{value:u(Float64Array)}),Object.defineProperty(o.prototype,"setUint8",{value:f(Uint8Array)}),Object.defineProperty(o.prototype,"setInt8",{value:f(Int8Array)}),Object.defineProperty(o.prototype,"setUint16",{value:f(Uint16Array)}),Object.defineProperty(o.prototype,"setInt16",{value:f(Int16Array)}),Object.defineProperty(o.prototype,"setUint32",{value:f(Uint32Array)}),Object.defineProperty(o.prototype,"setInt32",{value:f(Int32Array)}),Object.defineProperty(o.prototype,"setFloat32",{value:f(Float32Array)}),Object.defineProperty(o.prototype,"setFloat64",{value:f(Float64Array)}),e.DataView=e.DataView||o}()})(this);