webswingRequirejs.define(["jquery","webswing-util"],function(t,n){return function(){function s(e){var r="";i.cfg.anonym||(r='<span class="pull-right"><a href="javascript:;" data-id="logout">Logout</a></span>',r=r+'<h4 class="modal-title" id="myModalLabel">Hello <span>'+i.getUser()+"</span>. "),r+="Select your application:</h4>";var s={logout_click:function(){i.logout()},application_click:function(){var e=t(this).attr("data-name"),n=t(this).attr("data-applet"),r=t(this).attr("data-always-restart");i.startApplication(e,"true"===n,"true"===r)}},o;if(e==null||e.length===0)r=null,o="<p>Sorry, there is no application available for you.</p>";else if(i.cfg.applicationName!=null){var u=!1,a=!1,f=!1;e.forEach(function(e){e.name===i.cfg.applicationName&&(u=!0,a=e.applet,f=e.alwaysRestart)});if(u){i.cfg.mirror?i.startMirrorView(i.cfg.clientId,i.cfg.applicationName):i.startApplication(i.cfg.applicationName,a,f);return}r=null,o='<p>Sorry, application "'+i.cfg.applicationName+'" is not available for you.</p>'}else{o='<div class="row">';for(var l in e){var c=e[l];c.name==="adminConsoleApplicationName"?o+='<div class="col-xs-4 col-sm-3 col-md-2"><div class="thumbnail" style="max-width: 155px" onclick="window.location.href = \''+i.cfg.connectionUrl+'admin\';"><img src="'+i.cfg.connectionUrl+'admin/img/admin.png" class="img-thumbnail"/><div class="caption">Admin console</div></div></div>':o+='<div class="col-xs-4 col-sm-3 col-md-2"><div class="thumbnail" style="max-width: 155px" data-id="application" data-name="'+c.name+'" data-applet="'+c.applet+'" data-always-restart="'+c.alwaysRestart+'"><img src="'+n.getImageString(c.base64Icon)+'" class="img-thumbnail"/><div class="caption">'+c.name+"</div></div></div>"}o+="</div>"}i.showDialog({header:r,content:o,events:s})}function o(){i.hideDialog()}var r=this,i;r.injects=i={cfg:"webswing.config",getUser:"login.user",logout:"login.logout",startApplication:"base.startApplication",startMirrorView:"base.startMirrorView",showDialog:"dialog.show",hideDialog:"dialog.hide"},r.provides={show:s,hide:o}}});