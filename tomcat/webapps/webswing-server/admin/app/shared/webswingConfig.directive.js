(function(e){e(["text!shared/webswingConfig.template.html"],function(t){function n(){return{restrict:"E",template:t,scope:{config:"=",variables:"="},controllerAs:"vm",bindToController:!0,controller:r}}function r(e,t,n,r){function s(){i.config.applets==null&&(i.config.applets=[]),r.getDefault("applet").then(function(e){i.config.applets.push(e)})}function o(){i.config.applications==null&&(i.config.applications=[]),r.getDefault("application").then(function(e){i.config.applications.push(e)})}function u(){i.json=angular.toJson(i.config,!0)}function a(t){t.setReadOnly(i.readonly),e.$watch("vm.readonly",function(e){t.setReadOnly(e)}),e.$watch("vm.config",function(){u(),t.resize(!0)},!0)}function f(t,n){e.$watch(function(){var e=l(t,n);return e!==!1&&e!=="false"},function(e){i[t]=e})}function l(e,t){return n[e]!=null?n[e]:t}var i=this;i.readonly=f("readonly",!1),i.json=u(),i.aceLoaded=a,i.updateJson=u,i.newApp=o,i.newApplet=s,e.$watch("vm.json",function(e){try{var t=angular.fromJson(e);angular.merge(i.config,t)}catch(n){}},!0),e.$on("wsApplicationOpened",function(t,n,r){e.$broadcast("wsApplicationClose",n,r)})}return r.$inject=["$scope","$element","$attrs","configRestService"],n})})(adminConsole.define);