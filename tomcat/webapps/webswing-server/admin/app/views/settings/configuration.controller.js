(function(e){e([],function(){function t(e){function n(){e.getConfig().then(function(n){return angular.extend(t.config,n),e.getVariables()}).then(function(e){angular.extend(t.variables,e)})}function r(){n()}function i(){e.setConfig(t.config)}var t=this;t.config={},t.variables={},t.reset=r,t.apply=i,n()}return t.$inject=["configRestService"],t})})(adminConsole.define);