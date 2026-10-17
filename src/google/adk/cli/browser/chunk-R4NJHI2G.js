import{p as Ye}from"./chunk-TLHHCYOY.js";import{a as Je}from"./chunk-ROUF34MA.js";import{N as le,O as ue,T as de,U as fe,V as he,W as me,X as ke,Y as ye,Z as ge,_ as ot}from"./chunk-BTIP5KPZ.js";import{A as Se,B as Ce,C as Me,D as Ee,E as Ft,F as Ot,G as Ie,a as vt,c as oe,f as ce,g as rt,i as pe,j as ve,k as Te,l as be,r as xe,s as It,t as Yt,u as $t,v as Lt,w as At,x as we,y as _e,z as De}from"./chunk-JHJZJ2D5.js";import{a as l}from"./chunk-YUSHYV7C.js";import{d as wt,f as at}from"./chunk-BZGM3F3K.js";var $e=wt((Wt,Vt)=>{"use strict";(function(t,e){typeof Wt=="object"&&typeof Vt<"u"?Vt.exports=e():typeof define=="function"&&define.amd?define(e):(t=typeof globalThis<"u"?globalThis:t||self).dayjs_plugin_isoWeek=e()})(Wt,function(){"use strict";var t="day";return function(e,n,s){var r=function(w){return w.add(4-w.isoWeekday(),t)},f=n.prototype;f.isoWeekYear=function(){return r(this).year()},f.isoWeek=function(w){if(!this.$utils().u(w))return this.add(7*(w-this.isoWeek()),t);var S,N,Y,z,R=r(this),H=(S=this.isoWeekYear(),N=this.$u,Y=(N?s.utc:s)().year(S).startOf("year"),z=4-Y.isoWeekday(),Y.isoWeekday()>4&&(z+=7),Y.add(z,t));return R.diff(H,"week")+1},f.isoWeekday=function(w){return this.$utils().u(w)?this.day()||7:this.day(this.day()%7?w:w-7)};var y=f.startOf;f.startOf=function(w,S){var N=this.$utils(),Y=!!N.u(S)||S;return N.p(w)==="isoweek"?Y?this.date(this.date()-(this.isoWeekday()-1)).startOf("day"):this.date(this.date()-1-(this.isoWeekday()-1)+7).endOf("day"):y.bind(this)(w,S)}}})});var Le=wt((Pt,Nt)=>{"use strict";(function(t,e){typeof Pt=="object"&&typeof Nt<"u"?Nt.exports=e():typeof define=="function"&&define.amd?define(e):(t=typeof globalThis<"u"?globalThis:t||self).dayjs_plugin_customParseFormat=e()})(Pt,function(){"use strict";var t={LTS:"h:mm:ss A",LT:"h:mm A",L:"MM/DD/YYYY",LL:"MMMM D, YYYY",LLL:"MMMM D, YYYY h:mm A",LLLL:"dddd, MMMM D, YYYY h:mm A"},e=/(\[[^[]*\])|([-_:/.,()\s]+)|(A|a|Q|YYYY|YY?|ww?|MM?M?M?|Do|DD?|hh?|HH?|mm?|ss?|S{1,3}|z|ZZ?)/g,n=/\d/,s=/\d\d/,r=/\d\d?/,f=/\d*[^-_:/,()\s\d]+/,y={},w=function(x){return(x=+x)+(x>68?1900:2e3)},S=function(x){return function(_){this[x]=+_}},N=[/[+-]\d\d:?(\d\d)?|Z/,function(x){(this.zone||(this.zone={})).offset=(function(_){if(!_||_==="Z")return 0;var b=_.match(/([+-]|\d\d)/g),A=60*b[1]+(+b[2]||0);return A===0?0:b[0]==="+"?-A:A})(x)}],Y=function(x){var _=y[x];return _&&(_.indexOf?_:_.s.concat(_.f))},z=function(x,_){var b,A=y.meridiem;if(A){for(var X=1;X<=24;X+=1)if(x.indexOf(A(X,0,_))>-1){b=X>12;break}}else b=x===(_?"pm":"PM");return b},R={A:[f,function(x){this.afternoon=z(x,!1)}],a:[f,function(x){this.afternoon=z(x,!0)}],Q:[n,function(x){this.month=3*(x-1)+1}],S:[n,function(x){this.milliseconds=100*+x}],SS:[s,function(x){this.milliseconds=10*+x}],SSS:[/\d{3}/,function(x){this.milliseconds=+x}],s:[r,S("seconds")],ss:[r,S("seconds")],m:[r,S("minutes")],mm:[r,S("minutes")],H:[r,S("hours")],h:[r,S("hours")],HH:[r,S("hours")],hh:[r,S("hours")],D:[r,S("day")],DD:[s,S("day")],Do:[f,function(x){var _=y.ordinal,b=x.match(/\d+/);if(this.day=b[0],_)for(var A=1;A<=31;A+=1)_(A).replace(/\[|\]/g,"")===x&&(this.day=A)}],w:[r,S("week")],ww:[s,S("week")],M:[r,S("month")],MM:[s,S("month")],MMM:[f,function(x){var _=Y("months"),b=(Y("monthsShort")||_.map(function(A){return A.slice(0,3)})).indexOf(x)+1;if(b<1)throw new Error;this.month=b%12||b}],MMMM:[f,function(x){var _=Y("months").indexOf(x)+1;if(_<1)throw new Error;this.month=_%12||_}],Y:[/[+-]?\d+/,S("year")],YY:[s,function(x){this.year=w(x)}],YYYY:[/\d{4}/,S("year")],Z:N,ZZ:N};function H(x){var _,b;_=x,b=y&&y.formats;for(var A=(x=_.replace(/(\[[^\]]+])|(LTS?|l{1,4}|L{1,4})/g,function(W,F,k){var p=k&&k.toUpperCase();return F||b[k]||t[k]||b[p].replace(/(\[[^\]]+])|(MMMM|MM|DD|dddd)/g,function(v,T,a){return T||a.slice(1)})})).match(e),X=A.length,U=0;U<X;U+=1){var $=A[U],g=R[$],m=g&&g[0],I=g&&g[1];A[U]=I?{regex:m,parser:I}:$.replace(/^\[|\]$/g,"")}return function(W){for(var F={},k=0,p=0;k<X;k+=1){var v=A[k];if(typeof v=="string")p+=v.length;else{var T=v.regex,a=v.parser,h=W.slice(p),d=T.exec(h)[0];a.call(F,d),W=W.replace(d,"")}}return(function(u){var C=u.afternoon;if(C!==void 0){var i=u.hours;C?i<12&&(u.hours+=12):i===12&&(u.hours=0),delete u.afternoon}})(F),F}}return function(x,_,b){b.p.customParseFormat=!0,x&&x.parseTwoDigitYear&&(w=x.parseTwoDigitYear);var A=_.prototype,X=A.parse;A.parse=function(U){var $=U.date,g=U.utc,m=U.args;this.$u=g;var I=m[1];if(typeof I=="string"){var W=m[2]===!0,F=m[3]===!0,k=W||F,p=m[2];F&&(p=m[2]),y=this.$locale(),!W&&p&&(y=b.Ls[p]),this.$d=(function(h,d,u,C){try{if(["x","X"].indexOf(d)>-1)return new Date((d==="X"?1e3:1)*h);var i=H(d)(h),M=i.year,c=i.month,B=i.day,o=i.hours,D=i.minutes,E=i.seconds,V=i.milliseconds,P=i.zone,L=i.week,O=new Date,tt=B||(M||c?1:O.getDate()),et=M||O.getFullYear(),lt=0;M&&!c||(lt=c>0?c-1:O.getMonth());var ut,dt=o||0,j=D||0,nt=E||0,K=V||0;return P?new Date(Date.UTC(et,lt,tt,dt,j,nt,K+60*P.offset*1e3)):u?new Date(Date.UTC(et,lt,tt,dt,j,nt,K)):(ut=new Date(et,lt,tt,dt,j,nt,K),L&&(ut=C(ut).week(L).toDate()),ut)}catch(q){return new Date("")}})($,I,g,b),this.init(),p&&p!==!0&&(this.$L=this.locale(p).$L),k&&$!=this.format(I)&&(this.$d=new Date("")),y={}}else if(I instanceof Array)for(var v=I.length,T=1;T<=v;T+=1){m[1]=I[T-1];var a=b.apply(this,m);if(a.isValid()){this.$d=a.$d,this.$L=a.$L,this.init();break}T===v&&(this.$d=new Date(""))}else X.call(this,U)}}})});var Ae=wt((Rt,zt)=>{"use strict";(function(t,e){typeof Rt=="object"&&typeof zt<"u"?zt.exports=e():typeof define=="function"&&define.amd?define(e):(t=typeof globalThis<"u"?globalThis:t||self).dayjs_plugin_advancedFormat=e()})(Rt,function(){"use strict";return function(t,e){var n=e.prototype,s=n.format;n.format=function(r){var f=this,y=this.$locale();if(!this.isValid())return s.bind(this)(r);var w=this.$utils(),S=(r||"YYYY-MM-DDTHH:mm:ssZ").replace(/\[([^\]]+)]|Q|wo|ww|w|WW|W|zzz|z|gggg|GGGG|Do|X|x|k{1,2}|S/g,function(N){switch(N){case"Q":return Math.ceil((f.$M+1)/3);case"Do":return y.ordinal(f.$D);case"gggg":return f.weekYear();case"GGGG":return f.isoWeekYear();case"wo":return y.ordinal(f.week(),"W");case"w":case"ww":return w.s(f.week(),N==="w"?1:2,"0");case"W":case"WW":return w.s(f.isoWeek(),N==="W"?1:2,"0");case"k":case"kk":return w.s(String(f.$H===0?24:f.$H),N==="k"?1:2,"0");case"X":return Math.floor(f.$d.getTime()/1e3);case"x":return f.$d.getTime();case"z":return"["+f.offsetName()+"]";case"zzz":return"["+f.offsetName("long")+"]";default:return N}});return s.bind(this)(S)}}})});var Fe=wt((Ht,Bt)=>{"use strict";(function(t,e){typeof Ht=="object"&&typeof Bt<"u"?Bt.exports=e():typeof define=="function"&&define.amd?define(e):(t=typeof globalThis<"u"?globalThis:t||self).dayjs_plugin_duration=e()})(Ht,function(){"use strict";var t,e,n=1e3,s=6e4,r=36e5,f=864e5,y=31536e6,w=2628e6,S=/^(-|\+)?P(?:([-+]?[0-9,.]*)Y)?(?:([-+]?[0-9,.]*)M)?(?:([-+]?[0-9,.]*)W)?(?:([-+]?[0-9,.]*)D)?(?:T(?:([-+]?[0-9,.]*)H)?(?:([-+]?[0-9,.]*)M)?(?:([-+]?[0-9,.]*)S)?)?$/,N=/\[([^\]]+)]|YYYY|YY|Y|M{1,2}|D{1,2}|H{1,2}|m{1,2}|s{1,2}|SSS/g,Y={years:y,months:w,days:f,hours:r,minutes:s,seconds:n,milliseconds:1,weeks:6048e5},z=function($){return $ instanceof X},R=function($,g,m){return new X($,m,g.$l)},H=function($){return e.p($)+"s"},x=function($){return $<0},_=function($){return x($)?Math.ceil($):Math.floor($)},b=function($){return Math.abs($)},A=function($,g){return $?x($)?{negative:!0,format:""+b($)+g}:{negative:!1,format:""+$+g}:{negative:!1,format:""}},X=(function(){function $(m,I,W){var F=this;if(this.$d={},this.$l=W,m===void 0&&(this.$ms=0,this.parseFromMilliseconds()),I)return R(m*Y[H(I)],this);if(typeof m=="number")return this.$ms=m,this.parseFromMilliseconds(),this;if(typeof m=="object")return Object.keys(m).forEach(function(v){F.$d[H(v)]=m[v]}),this.calMilliseconds(),this;if(typeof m=="string"){var k=m.match(S);if(k){var p=k.slice(2).map(function(v){return v!=null?Number(v):0});return this.$d.years=p[0],this.$d.months=p[1],this.$d.weeks=p[2],this.$d.days=p[3],this.$d.hours=p[4],this.$d.minutes=p[5],this.$d.seconds=p[6],this.calMilliseconds(),this}}return this}var g=$.prototype;return g.calMilliseconds=function(){var m=this;this.$ms=Object.keys(this.$d).reduce(function(I,W){return I+(m.$d[W]||0)*Y[W]},0)},g.parseFromMilliseconds=function(){var m=this.$ms;this.$d.years=_(m/y),m%=y,this.$d.months=_(m/w),m%=w,this.$d.days=_(m/f),m%=f,this.$d.hours=_(m/r),m%=r,this.$d.minutes=_(m/s),m%=s,this.$d.seconds=_(m/n),m%=n,this.$d.milliseconds=m},g.toISOString=function(){var m=A(this.$d.years,"Y"),I=A(this.$d.months,"M"),W=+this.$d.days||0;this.$d.weeks&&(W+=7*this.$d.weeks);var F=A(W,"D"),k=A(this.$d.hours,"H"),p=A(this.$d.minutes,"M"),v=this.$d.seconds||0;this.$d.milliseconds&&(v+=this.$d.milliseconds/1e3,v=Math.round(1e3*v)/1e3);var T=A(v,"S"),a=m.negative||I.negative||F.negative||k.negative||p.negative||T.negative,h=k.format||p.format||T.format?"T":"",d=(a?"-":"")+"P"+m.format+I.format+F.format+h+k.format+p.format+T.format;return d==="P"||d==="-P"?"P0D":d},g.toJSON=function(){return this.toISOString()},g.format=function(m){var I=m||"YYYY-MM-DDTHH:mm:ss",W={Y:this.$d.years,YY:e.s(this.$d.years,2,"0"),YYYY:e.s(this.$d.years,4,"0"),M:this.$d.months,MM:e.s(this.$d.months,2,"0"),D:this.$d.days,DD:e.s(this.$d.days,2,"0"),H:this.$d.hours,HH:e.s(this.$d.hours,2,"0"),m:this.$d.minutes,mm:e.s(this.$d.minutes,2,"0"),s:this.$d.seconds,ss:e.s(this.$d.seconds,2,"0"),SSS:e.s(this.$d.milliseconds,3,"0")};return I.replace(N,function(F,k){return k||String(W[F])})},g.as=function(m){return this.$ms/Y[H(m)]},g.get=function(m){var I=this.$ms,W=H(m);return W==="milliseconds"?I%=1e3:I=W==="weeks"?_(I/Y[W]):this.$d[W],I||0},g.add=function(m,I,W){var F;return F=I?m*Y[H(I)]:z(m)?m.$ms:R(m,this).$ms,R(this.$ms+F*(W?-1:1),this)},g.subtract=function(m,I){return this.add(m,I,!0)},g.locale=function(m){var I=this.clone();return I.$l=m,I},g.clone=function(){return R(this.$ms,this)},g.humanize=function(m){return t().add(this.$ms,"ms").locale(this.$l).fromNow(!m)},g.valueOf=function(){return this.asMilliseconds()},g.milliseconds=function(){return this.get("milliseconds")},g.asMilliseconds=function(){return this.as("milliseconds")},g.seconds=function(){return this.get("seconds")},g.asSeconds=function(){return this.as("seconds")},g.minutes=function(){return this.get("minutes")},g.asMinutes=function(){return this.as("minutes")},g.hours=function(){return this.get("hours")},g.asHours=function(){return this.as("hours")},g.days=function(){return this.get("days")},g.asDays=function(){return this.as("days")},g.weeks=function(){return this.get("weeks")},g.asWeeks=function(){return this.as("weeks")},g.months=function(){return this.get("months")},g.asMonths=function(){return this.as("months")},g.years=function(){return this.get("years")},g.asYears=function(){return this.as("years")},$})(),U=function($,g,m){return $.add(g.years()*m,"y").add(g.months()*m,"M").add(g.days()*m,"d").add(g.hours()*m,"h").add(g.minutes()*m,"m").add(g.seconds()*m,"s").add(g.milliseconds()*m,"ms")};return function($,g,m){t=m,e=m().$utils(),m.duration=function(F,k){var p=m.locale();return R(F,{$l:p},k)},m.isDuration=z;var I=g.prototype.add,W=g.prototype.subtract;g.prototype.add=function(F,k){return z(F)?U(this,F,1):I.bind(this)(F,k)},g.prototype.subtract=function(F,k){return z(F)?U(this,F,-1):W.bind(this)(F,k)}}})});var Pe=at(Je(),1),Q=at(ce(),1),Ne=at($e(),1),Re=at(Le(),1),ze=at(Ae(),1),mt=at(ce(),1),Ke=at(Fe(),1);var Gt=(function(){var t=l(function(T,a,h,d){for(h=h||{},d=T.length;d--;h[T[d]]=a);return h},"o"),e=[6,8,10,12,13,14,15,16,17,18,20,21,22,23,24,25,26,27,28,29,30,31,33,35,36,38,40],n=[1,26],s=[1,27],r=[1,28],f=[1,29],y=[1,30],w=[1,31],S=[1,32],N=[1,33],Y=[1,34],z=[1,9],R=[1,10],H=[1,11],x=[1,12],_=[1,13],b=[1,14],A=[1,15],X=[1,16],U=[1,19],$=[1,20],g=[1,21],m=[1,22],I=[1,23],W=[1,25],F=[1,35],k={trace:l(function(){},"trace"),yy:{},symbols_:{error:2,start:3,gantt:4,document:5,EOF:6,line:7,SPACE:8,statement:9,NL:10,weekday:11,weekday_monday:12,weekday_tuesday:13,weekday_wednesday:14,weekday_thursday:15,weekday_friday:16,weekday_saturday:17,weekday_sunday:18,weekend:19,weekend_friday:20,weekend_saturday:21,dateFormat:22,inclusiveEndDates:23,topAxis:24,axisFormat:25,tickInterval:26,excludes:27,includes:28,todayMarker:29,title:30,acc_title:31,acc_title_value:32,acc_descr:33,acc_descr_value:34,acc_descr_multiline_value:35,section:36,clickStatement:37,taskTxt:38,taskData:39,click:40,callbackname:41,callbackargs:42,href:43,clickStatementDebug:44,$accept:0,$end:1},terminals_:{2:"error",4:"gantt",6:"EOF",8:"SPACE",10:"NL",12:"weekday_monday",13:"weekday_tuesday",14:"weekday_wednesday",15:"weekday_thursday",16:"weekday_friday",17:"weekday_saturday",18:"weekday_sunday",20:"weekend_friday",21:"weekend_saturday",22:"dateFormat",23:"inclusiveEndDates",24:"topAxis",25:"axisFormat",26:"tickInterval",27:"excludes",28:"includes",29:"todayMarker",30:"title",31:"acc_title",32:"acc_title_value",33:"acc_descr",34:"acc_descr_value",35:"acc_descr_multiline_value",36:"section",38:"taskTxt",39:"taskData",40:"click",41:"callbackname",42:"callbackargs",43:"href"},productions_:[0,[3,3],[5,0],[5,2],[7,2],[7,1],[7,1],[7,1],[11,1],[11,1],[11,1],[11,1],[11,1],[11,1],[11,1],[19,1],[19,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,1],[9,2],[9,2],[9,1],[9,1],[9,1],[9,2],[37,2],[37,3],[37,3],[37,4],[37,3],[37,4],[37,2],[44,2],[44,3],[44,3],[44,4],[44,3],[44,4],[44,2]],performAction:l(function(a,h,d,u,C,i,M){var c=i.length-1;switch(C){case 1:return i[c-1];case 2:this.$=[];break;case 3:i[c-1].push(i[c]),this.$=i[c-1];break;case 4:case 5:this.$=i[c];break;case 6:case 7:this.$=[];break;case 8:u.setWeekday("monday");break;case 9:u.setWeekday("tuesday");break;case 10:u.setWeekday("wednesday");break;case 11:u.setWeekday("thursday");break;case 12:u.setWeekday("friday");break;case 13:u.setWeekday("saturday");break;case 14:u.setWeekday("sunday");break;case 15:u.setWeekend("friday");break;case 16:u.setWeekend("saturday");break;case 17:u.setDateFormat(i[c].substr(11)),this.$=i[c].substr(11);break;case 18:u.enableInclusiveEndDates(),this.$=i[c].substr(18);break;case 19:u.TopAxis(),this.$=i[c].substr(8);break;case 20:u.setAxisFormat(i[c].substr(11)),this.$=i[c].substr(11);break;case 21:u.setTickInterval(i[c].substr(13)),this.$=i[c].substr(13);break;case 22:u.setExcludes(i[c].substr(9)),this.$=i[c].substr(9);break;case 23:u.setIncludes(i[c].substr(9)),this.$=i[c].substr(9);break;case 24:u.setTodayMarker(i[c].substr(12)),this.$=i[c].substr(12);break;case 27:u.setDiagramTitle(i[c].substr(6)),this.$=i[c].substr(6);break;case 28:this.$=i[c].trim(),u.setAccTitle(this.$);break;case 29:case 30:this.$=i[c].trim(),u.setAccDescription(this.$);break;case 31:u.addSection(i[c].substr(8)),this.$=i[c].substr(8);break;case 33:u.addTask(i[c-1],i[c]),this.$="task";break;case 34:this.$=i[c-1],u.setClickEvent(i[c-1],i[c],null);break;case 35:this.$=i[c-2],u.setClickEvent(i[c-2],i[c-1],i[c]);break;case 36:this.$=i[c-2],u.setClickEvent(i[c-2],i[c-1],null),u.setLink(i[c-2],i[c]);break;case 37:this.$=i[c-3],u.setClickEvent(i[c-3],i[c-2],i[c-1]),u.setLink(i[c-3],i[c]);break;case 38:this.$=i[c-2],u.setClickEvent(i[c-2],i[c],null),u.setLink(i[c-2],i[c-1]);break;case 39:this.$=i[c-3],u.setClickEvent(i[c-3],i[c-1],i[c]),u.setLink(i[c-3],i[c-2]);break;case 40:this.$=i[c-1],u.setLink(i[c-1],i[c]);break;case 41:case 47:this.$=i[c-1]+" "+i[c];break;case 42:case 43:case 45:this.$=i[c-2]+" "+i[c-1]+" "+i[c];break;case 44:case 46:this.$=i[c-3]+" "+i[c-2]+" "+i[c-1]+" "+i[c];break}},"anonymous"),table:[{3:1,4:[1,2]},{1:[3]},t(e,[2,2],{5:3}),{6:[1,4],7:5,8:[1,6],9:7,10:[1,8],11:17,12:n,13:s,14:r,15:f,16:y,17:w,18:S,19:18,20:N,21:Y,22:z,23:R,24:H,25:x,26:_,27:b,28:A,29:X,30:U,31:$,33:g,35:m,36:I,37:24,38:W,40:F},t(e,[2,7],{1:[2,1]}),t(e,[2,3]),{9:36,11:17,12:n,13:s,14:r,15:f,16:y,17:w,18:S,19:18,20:N,21:Y,22:z,23:R,24:H,25:x,26:_,27:b,28:A,29:X,30:U,31:$,33:g,35:m,36:I,37:24,38:W,40:F},t(e,[2,5]),t(e,[2,6]),t(e,[2,17]),t(e,[2,18]),t(e,[2,19]),t(e,[2,20]),t(e,[2,21]),t(e,[2,22]),t(e,[2,23]),t(e,[2,24]),t(e,[2,25]),t(e,[2,26]),t(e,[2,27]),{32:[1,37]},{34:[1,38]},t(e,[2,30]),t(e,[2,31]),t(e,[2,32]),{39:[1,39]},t(e,[2,8]),t(e,[2,9]),t(e,[2,10]),t(e,[2,11]),t(e,[2,12]),t(e,[2,13]),t(e,[2,14]),t(e,[2,15]),t(e,[2,16]),{41:[1,40],43:[1,41]},t(e,[2,4]),t(e,[2,28]),t(e,[2,29]),t(e,[2,33]),t(e,[2,34],{42:[1,42],43:[1,43]}),t(e,[2,40],{41:[1,44]}),t(e,[2,35],{43:[1,45]}),t(e,[2,36]),t(e,[2,38],{42:[1,46]}),t(e,[2,37]),t(e,[2,39])],defaultActions:{},parseError:l(function(a,h){if(h.recoverable)this.trace(a);else{var d=new Error(a);throw d.hash=h,d}},"parseError"),parse:l(function(a){var h=this,d=[0],u=[],C=[null],i=[],M=this.table,c="",B=0,o=0,D=0,E=2,V=1,P=i.slice.call(arguments,1),L=Object.create(this.lexer),O={yy:{}};for(var tt in this.yy)Object.prototype.hasOwnProperty.call(this.yy,tt)&&(O.yy[tt]=this.yy[tt]);L.setInput(a,O.yy),O.yy.lexer=L,O.yy.parser=this,typeof L.yylloc>"u"&&(L.yylloc={});var et=L.yylloc;i.push(et);var lt=L.options&&L.options.ranges;typeof O.yy.parseError=="function"?this.parseError=O.yy.parseError:this.parseError=Object.getPrototypeOf(this).parseError;function ut(Z){d.length=d.length-2*Z,C.length=C.length-Z,i.length=i.length-Z}l(ut,"popStack");function dt(){var Z;return Z=u.pop()||L.lex()||V,typeof Z!="number"&&(Z instanceof Array&&(u=Z,Z=u.pop()),Z=h.symbols_[Z]||Z),Z}l(dt,"lex");for(var j,nt,K,q,Bi,Mt,ft={},bt,it,ae,xt;;){if(K=d[d.length-1],this.defaultActions[K]?q=this.defaultActions[K]:((j===null||typeof j>"u")&&(j=dt()),q=M[K]&&M[K][j]),typeof q>"u"||!q.length||!q[0]){var Et="";xt=[];for(bt in M[K])this.terminals_[bt]&&bt>E&&xt.push("'"+this.terminals_[bt]+"'");L.showPosition?Et="Parse error on line "+(B+1)+`:
`+L.showPosition()+`
Expecting `+xt.join(", ")+", got '"+(this.terminals_[j]||j)+"'":Et="Parse error on line "+(B+1)+": Unexpected "+(j==V?"end of input":"'"+(this.terminals_[j]||j)+"'"),this.parseError(Et,{text:L.match,token:this.terminals_[j]||j,line:L.yylineno,loc:et,expected:xt})}if(q[0]instanceof Array&&q.length>1)throw new Error("Parse Error: multiple actions possible at state: "+K+", token: "+j);switch(q[0]){case 1:d.push(j),C.push(L.yytext),i.push(L.yylloc),d.push(q[1]),j=null,nt?(j=nt,nt=null):(o=L.yyleng,c=L.yytext,B=L.yylineno,et=L.yylloc,D>0&&D--);break;case 2:if(it=this.productions_[q[1]][1],ft.$=C[C.length-it],ft._$={first_line:i[i.length-(it||1)].first_line,last_line:i[i.length-1].last_line,first_column:i[i.length-(it||1)].first_column,last_column:i[i.length-1].last_column},lt&&(ft._$.range=[i[i.length-(it||1)].range[0],i[i.length-1].range[1]]),Mt=this.performAction.apply(ft,[c,o,B,O.yy,q[1],C,i].concat(P)),typeof Mt<"u")return Mt;it&&(d=d.slice(0,-1*it*2),C=C.slice(0,-1*it),i=i.slice(0,-1*it)),d.push(this.productions_[q[1]][0]),C.push(ft.$),i.push(ft._$),ae=M[d[d.length-2]][d[d.length-1]],d.push(ae);break;case 3:return!0}}return!0},"parse")},p=(function(){var T={EOF:1,parseError:l(function(h,d){if(this.yy.parser)this.yy.parser.parseError(h,d);else throw new Error(h)},"parseError"),setInput:l(function(a,h){return this.yy=h||this.yy||{},this._input=a,this._more=this._backtrack=this.done=!1,this.yylineno=this.yyleng=0,this.yytext=this.matched=this.match="",this.conditionStack=["INITIAL"],this.yylloc={first_line:1,first_column:0,last_line:1,last_column:0},this.options.ranges&&(this.yylloc.range=[0,0]),this.offset=0,this},"setInput"),input:l(function(){var a=this._input[0];this.yytext+=a,this.yyleng++,this.offset++,this.match+=a,this.matched+=a;var h=a.match(/(?:\r\n?|\n).*/g);return h?(this.yylineno++,this.yylloc.last_line++):this.yylloc.last_column++,this.options.ranges&&this.yylloc.range[1]++,this._input=this._input.slice(1),a},"input"),unput:l(function(a){var h=a.length,d=a.split(/(?:\r\n?|\n)/g);this._input=a+this._input,this.yytext=this.yytext.substr(0,this.yytext.length-h),this.offset-=h;var u=this.match.split(/(?:\r\n?|\n)/g);this.match=this.match.substr(0,this.match.length-1),this.matched=this.matched.substr(0,this.matched.length-1),d.length-1&&(this.yylineno-=d.length-1);var C=this.yylloc.range;return this.yylloc={first_line:this.yylloc.first_line,last_line:this.yylineno+1,first_column:this.yylloc.first_column,last_column:d?(d.length===u.length?this.yylloc.first_column:0)+u[u.length-d.length].length-d[0].length:this.yylloc.first_column-h},this.options.ranges&&(this.yylloc.range=[C[0],C[0]+this.yyleng-h]),this.yyleng=this.yytext.length,this},"unput"),more:l(function(){return this._more=!0,this},"more"),reject:l(function(){if(this.options.backtrack_lexer)this._backtrack=!0;else return this.parseError("Lexical error on line "+(this.yylineno+1)+`. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).
`+this.showPosition(),{text:"",token:null,line:this.yylineno});return this},"reject"),less:l(function(a){this.unput(this.match.slice(a))},"less"),pastInput:l(function(){var a=this.matched.substr(0,this.matched.length-this.match.length);return(a.length>20?"...":"")+a.substr(-20).replace(/\n/g,"")},"pastInput"),upcomingInput:l(function(){var a=this.match;return a.length<20&&(a+=this._input.substr(0,20-a.length)),(a.substr(0,20)+(a.length>20?"...":"")).replace(/\n/g,"")},"upcomingInput"),showPosition:l(function(){var a=this.pastInput(),h=new Array(a.length+1).join("-");return a+this.upcomingInput()+`
`+h+"^"},"showPosition"),test_match:l(function(a,h){var d,u,C;if(this.options.backtrack_lexer&&(C={yylineno:this.yylineno,yylloc:{first_line:this.yylloc.first_line,last_line:this.last_line,first_column:this.yylloc.first_column,last_column:this.yylloc.last_column},yytext:this.yytext,match:this.match,matches:this.matches,matched:this.matched,yyleng:this.yyleng,offset:this.offset,_more:this._more,_input:this._input,yy:this.yy,conditionStack:this.conditionStack.slice(0),done:this.done},this.options.ranges&&(C.yylloc.range=this.yylloc.range.slice(0))),u=a[0].match(/(?:\r\n?|\n).*/g),u&&(this.yylineno+=u.length),this.yylloc={first_line:this.yylloc.last_line,last_line:this.yylineno+1,first_column:this.yylloc.last_column,last_column:u?u[u.length-1].length-u[u.length-1].match(/\r?\n?/)[0].length:this.yylloc.last_column+a[0].length},this.yytext+=a[0],this.match+=a[0],this.matches=a,this.yyleng=this.yytext.length,this.options.ranges&&(this.yylloc.range=[this.offset,this.offset+=this.yyleng]),this._more=!1,this._backtrack=!1,this._input=this._input.slice(a[0].length),this.matched+=a[0],d=this.performAction.call(this,this.yy,this,h,this.conditionStack[this.conditionStack.length-1]),this.done&&this._input&&(this.done=!1),d)return d;if(this._backtrack){for(var i in C)this[i]=C[i];return!1}return!1},"test_match"),next:l(function(){if(this.done)return this.EOF;this._input||(this.done=!0);var a,h,d,u;this._more||(this.yytext="",this.match="");for(var C=this._currentRules(),i=0;i<C.length;i++)if(d=this._input.match(this.rules[C[i]]),d&&(!h||d[0].length>h[0].length)){if(h=d,u=i,this.options.backtrack_lexer){if(a=this.test_match(d,C[i]),a!==!1)return a;if(this._backtrack){h=!1;continue}else return!1}else if(!this.options.flex)break}return h?(a=this.test_match(h,C[u]),a!==!1?a:!1):this._input===""?this.EOF:this.parseError("Lexical error on line "+(this.yylineno+1)+`. Unrecognized text.
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"next"),lex:l(function(){var h=this.next();return h||this.lex()},"lex"),begin:l(function(h){this.conditionStack.push(h)},"begin"),popState:l(function(){var h=this.conditionStack.length-1;return h>0?this.conditionStack.pop():this.conditionStack[0]},"popState"),_currentRules:l(function(){return this.conditionStack.length&&this.conditionStack[this.conditionStack.length-1]?this.conditions[this.conditionStack[this.conditionStack.length-1]].rules:this.conditions.INITIAL.rules},"_currentRules"),topState:l(function(h){return h=this.conditionStack.length-1-Math.abs(h||0),h>=0?this.conditionStack[h]:"INITIAL"},"topState"),pushState:l(function(h){this.begin(h)},"pushState"),stateStackSize:l(function(){return this.conditionStack.length},"stateStackSize"),options:{"case-insensitive":!0},performAction:l(function(h,d,u,C){var i=C;switch(u){case 0:return this.begin("open_directive"),"open_directive";break;case 1:return this.begin("acc_title"),31;break;case 2:return this.popState(),"acc_title_value";break;case 3:return this.begin("acc_descr"),33;break;case 4:return this.popState(),"acc_descr_value";break;case 5:this.begin("acc_descr_multiline");break;case 6:this.popState();break;case 7:return"acc_descr_multiline_value";case 8:break;case 9:break;case 10:break;case 11:return 10;case 12:break;case 13:break;case 14:this.begin("href");break;case 15:this.popState();break;case 16:return 43;case 17:this.begin("callbackname");break;case 18:this.popState();break;case 19:this.popState(),this.begin("callbackargs");break;case 20:return 41;case 21:this.popState();break;case 22:return 42;case 23:this.begin("click");break;case 24:this.popState();break;case 25:return 40;case 26:return 4;case 27:return 22;case 28:return 23;case 29:return 24;case 30:return 25;case 31:return 26;case 32:return 28;case 33:return 27;case 34:return 29;case 35:return 12;case 36:return 13;case 37:return 14;case 38:return 15;case 39:return 16;case 40:return 17;case 41:return 18;case 42:return 20;case 43:return 21;case 44:return"date";case 45:return 30;case 46:return"accDescription";case 47:return 36;case 48:return 38;case 49:return 39;case 50:return":";case 51:return 6;case 52:return"INVALID"}},"anonymous"),rules:[/^(?:%%\{)/i,/^(?:accTitle\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*\{\s*)/i,/^(?:[\}])/i,/^(?:[^\}]*)/i,/^(?:%%(?!\{)*[^\n]*)/i,/^(?:[^\}]%%*[^\n]*)/i,/^(?:%%*[^\n]*[\n]*)/i,/^(?:[\n]+)/i,/^(?:\s+)/i,/^(?:%[^\n]*)/i,/^(?:href[\s]+["])/i,/^(?:["])/i,/^(?:[^"]*)/i,/^(?:call[\s]+)/i,/^(?:\([\s]*\))/i,/^(?:\()/i,/^(?:[^(]*)/i,/^(?:\))/i,/^(?:[^)]*)/i,/^(?:click[\s]+)/i,/^(?:[\s\n])/i,/^(?:[^\s\n]*)/i,/^(?:gantt\b)/i,/^(?:dateFormat\s[^#\n;]+)/i,/^(?:inclusiveEndDates\b)/i,/^(?:topAxis\b)/i,/^(?:axisFormat\s[^#\n;]+)/i,/^(?:tickInterval\s[^#\n;]+)/i,/^(?:includes\s[^#\n;]+)/i,/^(?:excludes\s[^#\n;]+)/i,/^(?:todayMarker\s[^\n;]+)/i,/^(?:weekday\s+monday\b)/i,/^(?:weekday\s+tuesday\b)/i,/^(?:weekday\s+wednesday\b)/i,/^(?:weekday\s+thursday\b)/i,/^(?:weekday\s+friday\b)/i,/^(?:weekday\s+saturday\b)/i,/^(?:weekday\s+sunday\b)/i,/^(?:weekend\s+friday\b)/i,/^(?:weekend\s+saturday\b)/i,/^(?:\d\d\d\d-\d\d-\d\d\b)/i,/^(?:title\s[^\n]+)/i,/^(?:accDescription\s[^#\n;]+)/i,/^(?:section\s[^\n]+)/i,/^(?:[^:\n]+)/i,/^(?::[^#\n;]+)/i,/^(?::)/i,/^(?:$)/i,/^(?:.)/i],conditions:{acc_descr_multiline:{rules:[6,7],inclusive:!1},acc_descr:{rules:[4],inclusive:!1},acc_title:{rules:[2],inclusive:!1},callbackargs:{rules:[21,22],inclusive:!1},callbackname:{rules:[18,19,20],inclusive:!1},href:{rules:[15,16],inclusive:!1},click:{rules:[24,25],inclusive:!1},INITIAL:{rules:[0,1,3,5,8,9,10,11,12,13,14,17,23,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52],inclusive:!0}}};return T})();k.lexer=p;function v(){this.yy={}}return l(v,"Parser"),v.prototype=k,k.Parser=v,new v})();Gt.parser=Gt;var ti=Gt;Q.default.extend(Ne.default);Q.default.extend(Re.default);Q.default.extend(ze.default);var Oe={friday:5,saturday:6},J="",Zt="",Qt=void 0,Kt="",yt=[],gt=[],Jt=new Map,te=[],St=[],pt="",ee="",He=["active","done","crit","milestone","vert"],ie=[],ht="",Tt=!1,se=!1,re="sunday",Ct="saturday",Xt=0,ei=l(function(){te=[],St=[],pt="",ie=[],_t=0,qt=void 0,Dt=void 0,G=[],J="",Zt="",ee="",Qt=void 0,Kt="",yt=[],gt=[],Tt=!1,se=!1,Xt=0,Jt=new Map,ht="",de(),re="sunday",Ct="saturday"},"clear"),ii=l(function(t){ht=t},"setDiagramId"),si=l(function(t){Zt=t},"setAxisFormat"),ri=l(function(){return Zt},"getAxisFormat"),ni=l(function(t){Qt=t},"setTickInterval"),ai=l(function(){return Qt},"getTickInterval"),oi=l(function(t){Kt=t},"setTodayMarker"),ci=l(function(){return Kt},"getTodayMarker"),li=l(function(t){J=t},"setDateFormat"),ui=l(function(){Tt=!0},"enableInclusiveEndDates"),di=l(function(){return Tt},"endDatesAreInclusive"),fi=l(function(){se=!0},"enableTopAxis"),hi=l(function(){return se},"topAxisEnabled"),mi=l(function(t){ee=t},"setDisplayMode"),ki=l(function(){return ee},"getDisplayMode"),yi=l(function(){return J},"getDateFormat"),Be=l((t,e)=>{let n=e.toLowerCase().split(/[\s,]+/).filter(s=>s!=="");return[...new Set([...t,...n])]},"mergeTokens"),gi=l(function(t){yt=Be(yt,t)},"setIncludes"),pi=l(function(){return yt},"getIncludes"),vi=l(function(t){gt=Be(gt,t)},"setExcludes"),Ti=l(function(){return gt},"getExcludes"),bi=l(function(){return Jt},"getLinks"),xi=l(function(t){pt=t,te.push(t)},"addSection"),wi=l(function(){return te},"getSections"),_i=l(function(){let t=We(),e=10,n=0;for(;!t&&n<e;)t=We(),n++;return St=G,St},"getTasks"),je=l(function(t,e,n,s){let r=t.format(e.trim()),f=t.format("YYYY-MM-DD");return s.includes(r)||s.includes(f)?!1:n.includes("weekends")&&(t.isoWeekday()===Oe[Ct]||t.isoWeekday()===Oe[Ct]+1)||n.includes(t.format("dddd").toLowerCase())?!0:n.includes(r)||n.includes(f)},"isInvalidDate"),Di=l(function(t){re=t},"setWeekday"),Si=l(function(){return re},"getWeekday"),Ci=l(function(t){Ct=t},"setWeekend"),Ge=l(function(t,e,n,s){if(!n.length||t.manualEndTime)return;let r;t.startTime instanceof Date?r=(0,Q.default)(t.startTime):r=(0,Q.default)(t.startTime,e,!0),r=r.add(1,"d");let f;t.endTime instanceof Date?f=(0,Q.default)(t.endTime):f=(0,Q.default)(t.endTime,e,!0);let[y,w]=Mi(r,f,e,n,s);t.endTime=y.toDate(),t.renderEndTime=w},"checkTaskDates"),Mi=l(function(t,e,n,s,r){let f=!1,y=null,w=e.add(1e4,"d");for(;t<=e;){if(f||(y=e.toDate()),f=je(t,n,s,r),f&&(e=e.add(1,"d"),e>w))throw new Error("Failed to find a valid date that was not excluded by `excludes` after 10,000 iterations.");t=t.add(1,"d")}return[e,y]},"fixTaskDates"),Ut=l(function(t,e,n){if(n=n.trim(),l(w=>{let S=w.trim();return S==="x"||S==="X"},"isTimestampFormat")(e)&&/^\d+$/.test(n))return new Date(Number(n));let f=/^after\s+(?<ids>[\d\w- ]+)/.exec(n);if(f!==null){let w=null;for(let N of f.groups.ids.split(" ")){let Y=ct(N);Y!==void 0&&(!w||Y.endTime>w.endTime)&&(w=Y)}if(w)return w.endTime;let S=new Date;return S.setHours(0,0,0,0),S}let y=(0,Q.default)(n,e.trim(),!0);if(y.isValid())return y.toDate();{rt.debug("Invalid date:"+n),rt.debug("With date format:"+e.trim());let w=new Date(n);if(w===void 0||isNaN(w.getTime())||w.getFullYear()<-1e4||w.getFullYear()>1e4)throw new Error("Invalid date:"+n);return w}},"getStartDate"),Xe=l(function(t){let e=/^(\d+(?:\.\d+)?)([Mdhmswy]|ms)$/.exec(t.trim());return e!==null?[Number.parseFloat(e[1]),e[2]]:[NaN,"ms"]},"parseDuration"),Ue=l(function(t,e,n,s=!1){n=n.trim();let f=/^until\s+(?<ids>[\d\w- ]+)/.exec(n);if(f!==null){let Y=null;for(let R of f.groups.ids.split(" ")){let H=ct(R);H!==void 0&&(!Y||H.startTime<Y.startTime)&&(Y=H)}if(Y)return Y.startTime;let z=new Date;return z.setHours(0,0,0,0),z}let y=(0,Q.default)(n,e.trim(),!0);if(y.isValid())return s&&(y=y.add(1,"d")),y.toDate();let w=(0,Q.default)(t),[S,N]=Xe(n);if(!Number.isNaN(S)){let Y=w.add(S,N);Y.isValid()&&(w=Y)}return w.toDate()},"getEndDate"),_t=0,kt=l(function(t){return t===void 0?(_t=_t+1,"task"+_t):t},"parseId"),Ei=l(function(t,e){let n;e.substr(0,1)===":"?n=e.substr(1,e.length):n=e;let s=n.split(","),r={};ne(s,r,He);for(let y=0;y<s.length;y++)s[y]=s[y].trim();let f="";switch(s.length){case 1:r.id=kt(),r.startTime=t.endTime,f=s[0];break;case 2:r.id=kt(),r.startTime=Ut(void 0,J,s[0]),f=s[1];break;case 3:r.id=kt(s[0]),r.startTime=Ut(void 0,J,s[1]),f=s[2];break;default:}return f&&(r.endTime=Ue(r.startTime,J,f,Tt),r.manualEndTime=(0,Q.default)(f,"YYYY-MM-DD",!0).isValid(),Ge(r,J,gt,yt)),r},"compileData"),Ii=l(function(t,e){let n;e.substr(0,1)===":"?n=e.substr(1,e.length):n=e;let s=n.split(","),r={};ne(s,r,He);for(let f=0;f<s.length;f++)s[f]=s[f].trim();switch(s.length){case 1:r.id=kt(),r.startTime={type:"prevTaskEnd",id:t},r.endTime={data:s[0]};break;case 2:r.id=kt(),r.startTime={type:"getStartDate",startData:s[0]},r.endTime={data:s[1]};break;case 3:r.id=kt(s[0]),r.startTime={type:"getStartDate",startData:s[1]},r.endTime={data:s[2]};break;default:}return r},"parseData"),qt,Dt,G=[],qe={},Yi=l(function(t,e){let n={section:pt,type:pt,processed:!1,manualEndTime:!1,renderEndTime:null,raw:{data:e},task:t,classes:[]},s=Ii(Dt,e);n.raw.startTime=s.startTime,n.raw.endTime=s.endTime,n.id=s.id,n.prevTaskId=Dt,n.active=s.active,n.done=s.done,n.crit=s.crit,n.milestone=s.milestone,n.vert=s.vert,n.vert?n.order=-1:(n.order=Xt,Xt++);let r=G.push(n);Dt=n.id,qe[n.id]=r-1},"addTask"),ct=l(function(t){let e=qe[t];return G[e]},"findTaskById"),$i=l(function(t,e){let n={section:pt,type:pt,description:t,task:t,classes:[]},s=Ei(qt,e);n.startTime=s.startTime,n.endTime=s.endTime,n.id=s.id,n.active=s.active,n.done=s.done,n.crit=s.crit,n.milestone=s.milestone,n.vert=s.vert,qt=n,St.push(n)},"addTaskOrg"),We=l(function(){let t=l(function(n){let s=G[n],r="";switch(G[n].raw.startTime.type){case"prevTaskEnd":{let f=ct(s.prevTaskId);s.startTime=f.endTime;break}case"getStartDate":r=Ut(void 0,J,G[n].raw.startTime.startData),r&&(G[n].startTime=r);break}return G[n].startTime&&(G[n].endTime=Ue(G[n].startTime,J,G[n].raw.endTime.data,Tt),G[n].endTime&&(G[n].processed=!0,G[n].manualEndTime=(0,Q.default)(G[n].raw.endTime.data,"YYYY-MM-DD",!0).isValid(),Ge(G[n],J,gt,yt))),G[n].processed},"compileTask"),e=!0;for(let[n,s]of G.entries())t(n),e=e&&s.processed;return e},"compileTasks"),Li=l(function(t,e){let n=e;ot().securityLevel!=="loose"&&(n=(0,Pe.sanitizeUrl)(e)),t.split(",").forEach(function(s){ct(s)!==void 0&&(Qe(s,()=>{window.open(n,"_self")}),Jt.set(s,n))}),Ze(t,"clickable")},"setLink"),Ze=l(function(t,e){t.split(",").forEach(function(n){let s=ct(n);s!==void 0&&s.classes.push(e)})},"setClass"),Ai=l(function(t,e,n){if(ot().securityLevel!=="loose"||e===void 0)return;let s=[];if(typeof n=="string"){s=n.split(/,(?=(?:(?:[^"]*"){2})*[^"]*$)/);for(let f=0;f<s.length;f++){let y=s[f].trim();y.startsWith('"')&&y.endsWith('"')&&(y=y.substr(1,y.length-2)),s[f]=y}}s.length===0&&s.push(t),ct(t)!==void 0&&Qe(t,()=>{Ye.runFunc(e,...s)})},"setClickFun"),Qe=l(function(t,e){ie.push(function(){let n=ht?`${ht}-${t}`:t,s=document.querySelector(`[id="${n}"]`);s!==null&&s.addEventListener("click",function(){e()})},function(){let n=ht?`${ht}-${t}`:t,s=document.querySelector(`[id="${n}-text"]`);s!==null&&s.addEventListener("click",function(){e()})})},"pushFun"),Fi=l(function(t,e,n){t.split(",").forEach(function(s){Ai(s,e,n)}),Ze(t,"clickable")},"setClickEvent"),Oi=l(function(t){ie.forEach(function(e){e(t)})},"bindFunctions"),Wi={getConfig:l(()=>ot().gantt,"getConfig"),clear:ei,setDateFormat:li,getDateFormat:yi,enableInclusiveEndDates:ui,endDatesAreInclusive:di,enableTopAxis:fi,topAxisEnabled:hi,setAxisFormat:si,getAxisFormat:ri,setTickInterval:ni,getTickInterval:ai,setTodayMarker:oi,getTodayMarker:ci,setAccTitle:fe,getAccTitle:he,setDiagramTitle:ye,getDiagramTitle:ge,setDiagramId:ii,setDisplayMode:mi,getDisplayMode:ki,setAccDescription:me,getAccDescription:ke,addSection:xi,getSections:wi,getTasks:_i,addTask:Yi,findTaskById:ct,addTaskOrg:$i,setIncludes:gi,getIncludes:pi,setExcludes:vi,getExcludes:Ti,setClickEvent:Fi,setLink:Li,getLinks:bi,bindFunctions:Oi,parseDuration:Xe,isInvalidDate:je,setWeekday:Di,getWeekday:Si,setWeekend:Ci};function ne(t,e,n){let s=!0;for(;s;)s=!1,n.forEach(function(r){let f="^\\s*"+r+"\\s*$",y=new RegExp(f);t[0].match(y)&&(e[r]=!0,t.shift(1),s=!0)})}l(ne,"getTaskTags");mt.default.extend(Ke.default);var Vi=l(function(){rt.debug("Something is calling, setConf, remove the call")},"setConf"),Ve={monday:_e,tuesday:De,wednesday:Se,thursday:Ce,friday:Me,saturday:Ee,sunday:we},Pi=l((t,e)=>{let n=[...t].map(()=>-1/0),s=[...t].sort((f,y)=>f.startTime-y.startTime||f.order-y.order),r=0;for(let f of s)for(let y=0;y<n.length;y++)if(f.startTime>=n[y]){n[y]=f.endTime,f.order=y+e,y>r&&(r=y);break}return r},"getMaxIntersections"),st,jt=1e4,Ni=l(function(t,e,n,s){let r=ot().gantt;s.db.setDiagramId(e);let f=ot().securityLevel,y;f==="sandbox"&&(y=vt("#i"+e));let w=f==="sandbox"?vt(y.nodes()[0].contentDocument.body):vt("body"),S=f==="sandbox"?y.nodes()[0].contentDocument:document,N=S.getElementById(e);st=N.parentElement.offsetWidth,st===void 0&&(st=1200),r.useWidth!==void 0&&(st=r.useWidth);let Y=s.db.getTasks(),z=Y.filter(k=>!k.vert),R=[];for(let k of z)R.push(k.type);R=F(R);let H={},x=2*r.topPadding;if(s.db.getDisplayMode()==="compact"||r.displayMode==="compact"){let k={};for(let v of z)k[v.section]===void 0?k[v.section]=[v]:k[v.section].push(v);let p=0;for(let v of Object.keys(k)){let T=Pi(k[v],p)+1;p+=T,x+=T*(r.barHeight+r.barGap),H[v]=T}}else{x+=z.length*(r.barHeight+r.barGap);for(let k of R)H[k]=z.filter(p=>p.type===k).length}N.setAttribute("viewBox","0 0 "+st+" "+x);let _=w.select(`[id="${e}"]`),b=Ie().domain([ve(Y,function(k){return k.startTime}),pe(Y,function(k){return k.endTime})]).rangeRound([0,st-r.leftPadding-r.rightPadding]);function A(k,p){let v=k.startTime,T=p.startTime,a=0;return v>T?a=1:v<T&&(a=-1),a}l(A,"taskCompare"),Y.sort(A),X(Y,st,x),ue(_,x,st,r.useMaxWidth),_.append("text").text(s.db.getDiagramTitle()).attr("x",st/2).attr("y",r.titleTopMargin).attr("class","titleText");function X(k,p,v){let T=r.barHeight,a=T+r.barGap,h=r.topPadding,d=r.leftPadding,u=xe().domain([0,R.length]).range(["#00B9FA","#F95002"]).interpolate(oe);$(a,h,d,p,v,k,s.db.getExcludes(),s.db.getIncludes()),m(d,h,p,v),U(k,a,h,d,T,u,p,v),I(a,h,d,T,u),W(d,h,p,v)}l(X,"makeGantt");function U(k,p,v,T,a,h,d){k.sort((o,D)=>o.vert===D.vert?0:o.vert?1:-1);let u=k.filter(o=>!o.vert),i=[...new Set(u.map(o=>o.order))].map(o=>u.find(D=>D.order===o));_.append("g").selectAll("rect").data(i).enter().append("rect").attr("x",0).attr("y",function(o,D){return D=o.order,D*p+v-2}).attr("width",function(){return d-r.rightPadding/2}).attr("height",p).attr("class",function(o){for(let[D,E]of R.entries())if(o.type===E)return"section section"+D%r.numberSectionStyles;return"section section0"}).enter();let M=_.append("g").selectAll("rect").data(k).enter(),c=s.db.getLinks();if(M.append("rect").attr("id",function(o){return e+"-"+o.id}).attr("rx",3).attr("ry",3).attr("x",function(o){return o.milestone?b(o.startTime)+T+.5*(b(o.endTime)-b(o.startTime))-.5*a:b(o.startTime)+T}).attr("y",function(o,D){return D=o.order,o.vert?r.gridLineStartPadding:D*p+v}).attr("width",function(o){return o.milestone?a:o.vert?.08*a:b(o.renderEndTime||o.endTime)-b(o.startTime)}).attr("height",function(o){return o.vert?u.length*(r.barHeight+r.barGap)+r.barHeight*2:a}).attr("transform-origin",function(o,D){return D=o.order,(b(o.startTime)+T+.5*(b(o.endTime)-b(o.startTime))).toString()+"px "+(D*p+v+.5*a).toString()+"px"}).attr("class",function(o){let D="task",E="";o.classes.length>0&&(E=o.classes.join(" "));let V=0;for(let[L,O]of R.entries())o.type===O&&(V=L%r.numberSectionStyles);let P="";return o.active?o.crit?P+=" activeCrit":P=" active":o.done?o.crit?P=" doneCrit":P=" done":o.crit&&(P+=" crit"),P.length===0&&(P=" task"),o.milestone&&(P=" milestone "+P),o.vert&&(P=" vert "+P),P+=V,P+=" "+E,D+P}),M.append("text").attr("id",function(o){return e+"-"+o.id+"-text"}).text(function(o){return o.task}).attr("font-size",r.fontSize).attr("x",function(o){let D=b(o.startTime),E=b(o.renderEndTime||o.endTime);if(o.milestone&&(D+=.5*(b(o.endTime)-b(o.startTime))-.5*a,E=D+a),o.vert)return b(o.startTime)+T;let V=this.getBBox().width;return V>E-D?E+V+1.5*r.leftPadding>d?D+T-5:E+T+5:(E-D)/2+D+T}).attr("y",function(o,D){return o.vert?r.gridLineStartPadding+u.length*(r.barHeight+r.barGap)+60:(D=o.order,D*p+r.barHeight/2+(r.fontSize/2-2)+v)}).attr("text-height",a).attr("class",function(o){let D=b(o.startTime),E=b(o.endTime);o.milestone&&(E=D+a);let V=this.getBBox().width,P="";o.classes.length>0&&(P=o.classes.join(" "));let L=0;for(let[tt,et]of R.entries())o.type===et&&(L=tt%r.numberSectionStyles);let O="";return o.active&&(o.crit?O="activeCritText"+L:O="activeText"+L),o.done?o.crit?O=O+" doneCritText"+L:O=O+" doneText"+L:o.crit&&(O=O+" critText"+L),o.milestone&&(O+=" milestoneText"),o.vert&&(O+=" vertText"),V>E-D?E+V+1.5*r.leftPadding>d?P+" taskTextOutsideLeft taskTextOutside"+L+" "+O:P+" taskTextOutsideRight taskTextOutside"+L+" "+O+" width-"+V:P+" taskText taskText"+L+" "+O+" width-"+V}),ot().securityLevel==="sandbox"){let o;o=vt("#i"+e);let D=o.nodes()[0].contentDocument;M.filter(function(E){return c.has(E.id)}).each(function(E){var V=D.querySelector("#"+CSS.escape(e+"-"+E.id)),P=D.querySelector("#"+CSS.escape(e+"-"+E.id+"-text"));let L=V.parentNode;var O=D.createElement("a");O.setAttribute("xlink:href",c.get(E.id)),O.setAttribute("target","_top"),L.appendChild(O),O.appendChild(V),O.appendChild(P)})}}l(U,"drawRects");function $(k,p,v,T,a,h,d,u){if(d.length===0&&u.length===0)return;let C,i;for(let{startTime:E,endTime:V}of h)(C===void 0||E<C)&&(C=E),(i===void 0||V>i)&&(i=V);if(!C||!i)return;if((0,mt.default)(i).diff((0,mt.default)(C),"year")>5){rt.warn("The difference between the min and max time is more than 5 years. This will cause performance issues. Skipping drawing exclude days.");return}let M=s.db.getDateFormat(),c=[],B=null,o=(0,mt.default)(C);for(;o.valueOf()<=i;)s.db.isInvalidDate(o,M,d,u)?B?B.end=o:B={start:o,end:o}:B&&(c.push(B),B=null),o=o.add(1,"d");_.append("g").selectAll("rect").data(c).enter().append("rect").attr("id",E=>e+"-exclude-"+E.start.format("YYYY-MM-DD")).attr("x",E=>b(E.start.startOf("day"))+v).attr("y",r.gridLineStartPadding).attr("width",E=>b(E.end.endOf("day"))-b(E.start.startOf("day"))).attr("height",a-p-r.gridLineStartPadding).attr("transform-origin",function(E,V){return(b(E.start)+v+.5*(b(E.end)-b(E.start))).toString()+"px "+(V*k+.5*a).toString()+"px"}).attr("class","exclude-range")}l($,"drawExcludeDays");function g(k,p,v,T){if(v<=0||k>p)return 1/0;let a=p-k,h=mt.default.duration({[T??"day"]:v}).asMilliseconds();return h<=0?1/0:Math.ceil(a/h)}l(g,"getEstimatedTickCount");function m(k,p,v,T){let a=s.db.getDateFormat(),h=s.db.getAxisFormat(),d;h?d=h:a==="D"?d="%d":d=r.axisFormat??"%Y-%m-%d";let u=be(b).tickSize(-T+p+r.gridLineStartPadding).tickFormat(Ot(d)),i=/^([1-9]\d*)(millisecond|second|minute|hour|day|week|month)$/.exec(s.db.getTickInterval()||r.tickInterval);if(i!==null){let M=parseInt(i[1],10);if(isNaN(M)||M<=0)rt.warn(`Invalid tick interval value: "${i[1]}". Skipping custom tick interval.`);else{let c=i[2],B=s.db.getWeekday()||r.weekday,o=b.domain(),D=o[0],E=o[1],V=g(D,E,M,c);if(V>jt)rt.warn(`The tick interval "${M}${c}" would generate ${V} ticks, which exceeds the maximum allowed (${jt}). This may indicate an invalid date or time range. Skipping custom tick interval.`);else switch(c){case"millisecond":u.ticks(It.every(M));break;case"second":u.ticks(Yt.every(M));break;case"minute":u.ticks($t.every(M));break;case"hour":u.ticks(Lt.every(M));break;case"day":u.ticks(At.every(M));break;case"week":u.ticks(Ve[B].every(M));break;case"month":u.ticks(Ft.every(M));break}}}if(_.append("g").attr("class","grid").attr("transform","translate("+k+", "+(T-50)+")").call(u).selectAll("text").style("text-anchor","middle").attr("fill","#000").attr("stroke","none").attr("font-size",10).attr("dy","1em"),s.db.topAxisEnabled()||r.topAxis){let M=Te(b).tickSize(-T+p+r.gridLineStartPadding).tickFormat(Ot(d));if(i!==null){let c=parseInt(i[1],10);if(isNaN(c)||c<=0)rt.warn(`Invalid tick interval value: "${i[1]}". Skipping custom tick interval.`);else{let B=i[2],o=s.db.getWeekday()||r.weekday,D=b.domain(),E=D[0],V=D[1];if(g(E,V,c,B)<=jt)switch(B){case"millisecond":M.ticks(It.every(c));break;case"second":M.ticks(Yt.every(c));break;case"minute":M.ticks($t.every(c));break;case"hour":M.ticks(Lt.every(c));break;case"day":M.ticks(At.every(c));break;case"week":M.ticks(Ve[o].every(c));break;case"month":M.ticks(Ft.every(c));break}}}_.append("g").attr("class","grid").attr("transform","translate("+k+", "+p+")").call(M).selectAll("text").style("text-anchor","middle").attr("fill","#000").attr("stroke","none").attr("font-size",10)}}l(m,"makeGrid");function I(k,p){let v=0,T=Object.keys(H).map(a=>[a,H[a]]);_.append("g").selectAll("text").data(T).enter().append(function(a){let h=a[0].split(le.lineBreakRegex),d=-(h.length-1)/2,u=S.createElementNS("http://www.w3.org/2000/svg","text");u.setAttribute("dy",d+"em");for(let[C,i]of h.entries()){let M=S.createElementNS("http://www.w3.org/2000/svg","tspan");M.setAttribute("alignment-baseline","central"),M.setAttribute("x","10"),C>0&&M.setAttribute("dy","1em"),M.textContent=i,u.appendChild(M)}return u}).attr("x",10).attr("y",function(a,h){if(h>0)for(let d=0;d<h;d++)return v+=T[h-1][1],a[1]*k/2+v*k+p;else return a[1]*k/2+p}).attr("font-size",r.sectionFontSize).attr("class",function(a){for(let[h,d]of R.entries())if(a[0]===d)return"sectionTitle sectionTitle"+h%r.numberSectionStyles;return"sectionTitle"})}l(I,"vertLabels");function W(k,p,v,T){let a=s.db.getTodayMarker();if(a==="off")return;let h=_.append("g").attr("class","today"),d=new Date,u=h.append("line");u.attr("x1",b(d)+k).attr("x2",b(d)+k).attr("y1",r.titleTopMargin).attr("y2",T-r.titleTopMargin).attr("class","today"),a!==""&&u.attr("style",a.replace(/,/g,";"))}l(W,"drawToday");function F(k){let p={},v=[];for(let T=0,a=k.length;T<a;++T)Object.prototype.hasOwnProperty.call(p,k[T])||(p[k[T]]=!0,v.push(k[T]));return v}l(F,"checkUnique")},"draw"),Ri={setConf:Vi,draw:Ni},zi=l(t=>`
  .mermaid-main-font {
        font-family: ${t.fontFamily};
  }

  .exclude-range {
    fill: ${t.excludeBkgColor};
  }

  .section {
    stroke: none;
    opacity: 0.2;
  }

  .section0 {
    fill: ${t.sectionBkgColor};
  }

  .section2 {
    fill: ${t.sectionBkgColor2};
  }

  .section1,
  .section3 {
    fill: ${t.altSectionBkgColor};
    opacity: 0.2;
  }

  .sectionTitle0 {
    fill: ${t.titleColor};
  }

  .sectionTitle1 {
    fill: ${t.titleColor};
  }

  .sectionTitle2 {
    fill: ${t.titleColor};
  }

  .sectionTitle3 {
    fill: ${t.titleColor};
  }

  .sectionTitle {
    text-anchor: start;
    font-family: ${t.fontFamily};
  }


  /* Grid and axis */

  .grid .tick {
    stroke: ${t.gridColor};
    opacity: 0.8;
    shape-rendering: crispEdges;
  }

  .grid .tick text {
    font-family: ${t.fontFamily};
    fill: ${t.textColor};
  }

  .grid path {
    stroke-width: 0;
  }


  /* Today line */

  .today {
    fill: none;
    stroke: ${t.todayLineColor};
    stroke-width: 2px;
  }


  /* Task styling */

  /* Default task */

  .task {
    stroke-width: 2;
  }

  .taskText {
    text-anchor: middle;
    font-family: ${t.fontFamily};
  }

  .taskTextOutsideRight {
    fill: ${t.taskTextDarkColor};
    text-anchor: start;
    font-family: ${t.fontFamily};
  }

  .taskTextOutsideLeft {
    fill: ${t.taskTextDarkColor};
    text-anchor: end;
  }


  /* Special case clickable */

  .task.clickable {
    cursor: pointer;
  }

  .taskText.clickable {
    cursor: pointer;
    fill: ${t.taskTextClickableColor} !important;
    font-weight: bold;
  }

  .taskTextOutsideLeft.clickable {
    cursor: pointer;
    fill: ${t.taskTextClickableColor} !important;
    font-weight: bold;
  }

  .taskTextOutsideRight.clickable {
    cursor: pointer;
    fill: ${t.taskTextClickableColor} !important;
    font-weight: bold;
  }


  /* Specific task settings for the sections*/

  .taskText0,
  .taskText1,
  .taskText2,
  .taskText3 {
    fill: ${t.taskTextColor};
  }

  .task0,
  .task1,
  .task2,
  .task3 {
    fill: ${t.taskBkgColor};
    stroke: ${t.taskBorderColor};
  }

  .taskTextOutside0,
  .taskTextOutside2
  {
    fill: ${t.taskTextOutsideColor};
  }

  .taskTextOutside1,
  .taskTextOutside3 {
    fill: ${t.taskTextOutsideColor};
  }


  /* Active task */

  .active0,
  .active1,
  .active2,
  .active3 {
    fill: ${t.activeTaskBkgColor};
    stroke: ${t.activeTaskBorderColor};
  }

  .activeText0,
  .activeText1,
  .activeText2,
  .activeText3 {
    fill: ${t.taskTextDarkColor} !important;
  }


  /* Completed task */

  .done0,
  .done1,
  .done2,
  .done3 {
    stroke: ${t.doneTaskBorderColor};
    fill: ${t.doneTaskBkgColor};
    stroke-width: 2;
  }

  .doneText0,
  .doneText1,
  .doneText2,
  .doneText3 {
    fill: ${t.taskTextDarkColor} !important;
  }

  /* Done task text displayed outside the bar sits against the diagram background,
     not against the done-task bar, so it must use the outside/contrast color. */
  .doneText0.taskTextOutsideLeft,
  .doneText0.taskTextOutsideRight,
  .doneText1.taskTextOutsideLeft,
  .doneText1.taskTextOutsideRight,
  .doneText2.taskTextOutsideLeft,
  .doneText2.taskTextOutsideRight,
  .doneText3.taskTextOutsideLeft,
  .doneText3.taskTextOutsideRight {
    fill: ${t.taskTextOutsideColor} !important;
  }


  /* Tasks on the critical line */

  .crit0,
  .crit1,
  .crit2,
  .crit3 {
    stroke: ${t.critBorderColor};
    fill: ${t.critBkgColor};
    stroke-width: 2;
  }

  .activeCrit0,
  .activeCrit1,
  .activeCrit2,
  .activeCrit3 {
    stroke: ${t.critBorderColor};
    fill: ${t.activeTaskBkgColor};
    stroke-width: 2;
  }

  .doneCrit0,
  .doneCrit1,
  .doneCrit2,
  .doneCrit3 {
    stroke: ${t.critBorderColor};
    fill: ${t.doneTaskBkgColor};
    stroke-width: 2;
    cursor: pointer;
    shape-rendering: crispEdges;
  }

  .milestone {
    transform: rotate(45deg) scale(0.8,0.8);
  }

  .milestoneText {
    font-style: italic;
  }
  .doneCritText0,
  .doneCritText1,
  .doneCritText2,
  .doneCritText3 {
    fill: ${t.taskTextDarkColor} !important;
  }

  /* Done-crit task text outside the bar \u2014 same reasoning as doneText above. */
  .doneCritText0.taskTextOutsideLeft,
  .doneCritText0.taskTextOutsideRight,
  .doneCritText1.taskTextOutsideLeft,
  .doneCritText1.taskTextOutsideRight,
  .doneCritText2.taskTextOutsideLeft,
  .doneCritText2.taskTextOutsideRight,
  .doneCritText3.taskTextOutsideLeft,
  .doneCritText3.taskTextOutsideRight {
    fill: ${t.taskTextOutsideColor} !important;
  }

  .vert {
    stroke: ${t.vertLineColor};
  }

  .vertText {
    font-size: 15px;
    text-anchor: middle;
    fill: ${t.vertLineColor} !important;
  }

  .activeCritText0,
  .activeCritText1,
  .activeCritText2,
  .activeCritText3 {
    fill: ${t.taskTextDarkColor} !important;
  }

  .titleText {
    text-anchor: middle;
    font-size: 18px;
    fill: ${t.titleColor||t.textColor};
    font-family: ${t.fontFamily};
  }
`,"getStyles"),Hi=zi,Zi={parser:ti,db:Wi,renderer:Ri,styles:Hi};export{Zi as diagram};
