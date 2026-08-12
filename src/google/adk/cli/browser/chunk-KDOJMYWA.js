import{a as mt}from"./chunk-B2DSW4QB.js";import{a as ft,b as pt,c as gt,f as Z}from"./chunk-LT3WOUUJ.js";import"./chunk-GP6TCC26.js";import{N as at,R as lt,S as ot,T as ct,U as ht,V as ut,W as yt,X as dt,Y as F}from"./chunk-37QI3DOO.js";import{K as U,a as W,g as s}from"./chunk-JRNAXTJ7.js";import"./chunk-RMXJBC7V.js";var K=(function(){var t=s(function(h,i,n,l){for(n=n||{},l=h.length;l--;n[h[l]]=i);return n},"o"),e=[6,8,10,11,12,14,16,17,18],a=[1,9],f=[1,10],r=[1,11],u=[1,12],p=[1,13],o=[1,14],g={trace:s(function(){},"trace"),yy:{},symbols_:{error:2,start:3,journey:4,document:5,EOF:6,line:7,SPACE:8,statement:9,NEWLINE:10,title:11,acc_title:12,acc_title_value:13,acc_descr:14,acc_descr_value:15,acc_descr_multiline_value:16,section:17,taskName:18,taskData:19,$accept:0,$end:1},terminals_:{2:"error",4:"journey",6:"EOF",8:"SPACE",10:"NEWLINE",11:"title",12:"acc_title",13:"acc_title_value",14:"acc_descr",15:"acc_descr_value",16:"acc_descr_multiline_value",17:"section",18:"taskName",19:"taskData"},productions_:[0,[3,3],[5,0],[5,2],[7,2],[7,1],[7,1],[7,1],[9,1],[9,2],[9,2],[9,1],[9,1],[9,2]],performAction:s(function(i,n,l,y,d,c,_){var k=c.length-1;switch(d){case 1:return c[k-1];case 2:this.$=[];break;case 3:c[k-1].push(c[k]),this.$=c[k-1];break;case 4:case 5:this.$=c[k];break;case 6:case 7:this.$=[];break;case 8:y.setDiagramTitle(c[k].substr(6)),this.$=c[k].substr(6);break;case 9:this.$=c[k].trim(),y.setAccTitle(this.$);break;case 10:case 11:this.$=c[k].trim(),y.setAccDescription(this.$);break;case 12:y.addSection(c[k].substr(8)),this.$=c[k].substr(8);break;case 13:y.addTask(c[k-1],c[k]),this.$="task";break}},"anonymous"),table:[{3:1,4:[1,2]},{1:[3]},t(e,[2,2],{5:3}),{6:[1,4],7:5,8:[1,6],9:7,10:[1,8],11:a,12:f,14:r,16:u,17:p,18:o},t(e,[2,7],{1:[2,1]}),t(e,[2,3]),{9:15,11:a,12:f,14:r,16:u,17:p,18:o},t(e,[2,5]),t(e,[2,6]),t(e,[2,8]),{13:[1,16]},{15:[1,17]},t(e,[2,11]),t(e,[2,12]),{19:[1,18]},t(e,[2,4]),t(e,[2,9]),t(e,[2,10]),t(e,[2,13])],defaultActions:{},parseError:s(function(i,n){if(n.recoverable)this.trace(i);else{var l=new Error(i);throw l.hash=n,l}},"parseError"),parse:s(function(i){var n=this,l=[0],y=[],d=[null],c=[],_=this.table,k="",C=0,et=0,rt=0,$t=2,it=1,Mt=c.slice.call(arguments,1),b=Object.create(this.lexer),I={yy:{}};for(var Y in this.yy)Object.prototype.hasOwnProperty.call(this.yy,Y)&&(I.yy[Y]=this.yy[Y]);b.setInput(i,I.yy),I.yy.lexer=b,I.yy.parser=this,typeof b.yylloc>"u"&&(b.yylloc={});var q=b.yylloc;c.push(q);var Et=b.options&&b.options.ranges;typeof I.yy.parseError=="function"?this.parseError=I.yy.parseError:this.parseError=Object.getPrototypeOf(this).parseError;function Ct(w){l.length=l.length-2*w,d.length=d.length-w,c.length=c.length-w}s(Ct,"popStack");function nt(){var w;return w=y.pop()||b.lex()||it,typeof w!="number"&&(w instanceof Array&&(y=w,w=y.pop()),w=n.symbols_[w]||w),w}s(nt,"lex");for(var v,H,A,T,Jt,X,V={},N,M,st,z;;){if(A=l[l.length-1],this.defaultActions[A]?T=this.defaultActions[A]:((v===null||typeof v>"u")&&(v=nt()),T=_[A]&&_[A][v]),typeof T>"u"||!T.length||!T[0]){var G="";z=[];for(N in _[A])this.terminals_[N]&&N>$t&&z.push("'"+this.terminals_[N]+"'");b.showPosition?G="Parse error on line "+(C+1)+`:
`+b.showPosition()+`
Expecting `+z.join(", ")+", got '"+(this.terminals_[v]||v)+"'":G="Parse error on line "+(C+1)+": Unexpected "+(v==it?"end of input":"'"+(this.terminals_[v]||v)+"'"),this.parseError(G,{text:b.match,token:this.terminals_[v]||v,line:b.yylineno,loc:q,expected:z})}if(T[0]instanceof Array&&T.length>1)throw new Error("Parse Error: multiple actions possible at state: "+A+", token: "+v);switch(T[0]){case 1:l.push(v),d.push(b.yytext),c.push(b.yylloc),l.push(T[1]),v=null,H?(v=H,H=null):(et=b.yyleng,k=b.yytext,C=b.yylineno,q=b.yylloc,rt>0&&rt--);break;case 2:if(M=this.productions_[T[1]][1],V.$=d[d.length-M],V._$={first_line:c[c.length-(M||1)].first_line,last_line:c[c.length-1].last_line,first_column:c[c.length-(M||1)].first_column,last_column:c[c.length-1].last_column},Et&&(V._$.range=[c[c.length-(M||1)].range[0],c[c.length-1].range[1]]),X=this.performAction.apply(V,[k,et,C,I.yy,T[1],d,c].concat(Mt)),typeof X<"u")return X;M&&(l=l.slice(0,-1*M*2),d=d.slice(0,-1*M),c=c.slice(0,-1*M)),l.push(this.productions_[T[1]][0]),d.push(V.$),c.push(V._$),st=_[l[l.length-2]][l[l.length-1]],l.push(st);break;case 3:return!0}}return!0},"parse")},m=(function(){var h={EOF:1,parseError:s(function(n,l){if(this.yy.parser)this.yy.parser.parseError(n,l);else throw new Error(n)},"parseError"),setInput:s(function(i,n){return this.yy=n||this.yy||{},this._input=i,this._more=this._backtrack=this.done=!1,this.yylineno=this.yyleng=0,this.yytext=this.matched=this.match="",this.conditionStack=["INITIAL"],this.yylloc={first_line:1,first_column:0,last_line:1,last_column:0},this.options.ranges&&(this.yylloc.range=[0,0]),this.offset=0,this},"setInput"),input:s(function(){var i=this._input[0];this.yytext+=i,this.yyleng++,this.offset++,this.match+=i,this.matched+=i;var n=i.match(/(?:\r\n?|\n).*/g);return n?(this.yylineno++,this.yylloc.last_line++):this.yylloc.last_column++,this.options.ranges&&this.yylloc.range[1]++,this._input=this._input.slice(1),i},"input"),unput:s(function(i){var n=i.length,l=i.split(/(?:\r\n?|\n)/g);this._input=i+this._input,this.yytext=this.yytext.substr(0,this.yytext.length-n),this.offset-=n;var y=this.match.split(/(?:\r\n?|\n)/g);this.match=this.match.substr(0,this.match.length-1),this.matched=this.matched.substr(0,this.matched.length-1),l.length-1&&(this.yylineno-=l.length-1);var d=this.yylloc.range;return this.yylloc={first_line:this.yylloc.first_line,last_line:this.yylineno+1,first_column:this.yylloc.first_column,last_column:l?(l.length===y.length?this.yylloc.first_column:0)+y[y.length-l.length].length-l[0].length:this.yylloc.first_column-n},this.options.ranges&&(this.yylloc.range=[d[0],d[0]+this.yyleng-n]),this.yyleng=this.yytext.length,this},"unput"),more:s(function(){return this._more=!0,this},"more"),reject:s(function(){if(this.options.backtrack_lexer)this._backtrack=!0;else return this.parseError("Lexical error on line "+(this.yylineno+1)+`. You can only invoke reject() in the lexer when the lexer is of the backtracking persuasion (options.backtrack_lexer = true).
`+this.showPosition(),{text:"",token:null,line:this.yylineno});return this},"reject"),less:s(function(i){this.unput(this.match.slice(i))},"less"),pastInput:s(function(){var i=this.matched.substr(0,this.matched.length-this.match.length);return(i.length>20?"...":"")+i.substr(-20).replace(/\n/g,"")},"pastInput"),upcomingInput:s(function(){var i=this.match;return i.length<20&&(i+=this._input.substr(0,20-i.length)),(i.substr(0,20)+(i.length>20?"...":"")).replace(/\n/g,"")},"upcomingInput"),showPosition:s(function(){var i=this.pastInput(),n=new Array(i.length+1).join("-");return i+this.upcomingInput()+`
`+n+"^"},"showPosition"),test_match:s(function(i,n){var l,y,d;if(this.options.backtrack_lexer&&(d={yylineno:this.yylineno,yylloc:{first_line:this.yylloc.first_line,last_line:this.last_line,first_column:this.yylloc.first_column,last_column:this.yylloc.last_column},yytext:this.yytext,match:this.match,matches:this.matches,matched:this.matched,yyleng:this.yyleng,offset:this.offset,_more:this._more,_input:this._input,yy:this.yy,conditionStack:this.conditionStack.slice(0),done:this.done},this.options.ranges&&(d.yylloc.range=this.yylloc.range.slice(0))),y=i[0].match(/(?:\r\n?|\n).*/g),y&&(this.yylineno+=y.length),this.yylloc={first_line:this.yylloc.last_line,last_line:this.yylineno+1,first_column:this.yylloc.last_column,last_column:y?y[y.length-1].length-y[y.length-1].match(/\r?\n?/)[0].length:this.yylloc.last_column+i[0].length},this.yytext+=i[0],this.match+=i[0],this.matches=i,this.yyleng=this.yytext.length,this.options.ranges&&(this.yylloc.range=[this.offset,this.offset+=this.yyleng]),this._more=!1,this._backtrack=!1,this._input=this._input.slice(i[0].length),this.matched+=i[0],l=this.performAction.call(this,this.yy,this,n,this.conditionStack[this.conditionStack.length-1]),this.done&&this._input&&(this.done=!1),l)return l;if(this._backtrack){for(var c in d)this[c]=d[c];return!1}return!1},"test_match"),next:s(function(){if(this.done)return this.EOF;this._input||(this.done=!0);var i,n,l,y;this._more||(this.yytext="",this.match="");for(var d=this._currentRules(),c=0;c<d.length;c++)if(l=this._input.match(this.rules[d[c]]),l&&(!n||l[0].length>n[0].length)){if(n=l,y=c,this.options.backtrack_lexer){if(i=this.test_match(l,d[c]),i!==!1)return i;if(this._backtrack){n=!1;continue}else return!1}else if(!this.options.flex)break}return n?(i=this.test_match(n,d[y]),i!==!1?i:!1):this._input===""?this.EOF:this.parseError("Lexical error on line "+(this.yylineno+1)+`. Unrecognized text.
`+this.showPosition(),{text:"",token:null,line:this.yylineno})},"next"),lex:s(function(){var n=this.next();return n||this.lex()},"lex"),begin:s(function(n){this.conditionStack.push(n)},"begin"),popState:s(function(){var n=this.conditionStack.length-1;return n>0?this.conditionStack.pop():this.conditionStack[0]},"popState"),_currentRules:s(function(){return this.conditionStack.length&&this.conditionStack[this.conditionStack.length-1]?this.conditions[this.conditionStack[this.conditionStack.length-1]].rules:this.conditions.INITIAL.rules},"_currentRules"),topState:s(function(n){return n=this.conditionStack.length-1-Math.abs(n||0),n>=0?this.conditionStack[n]:"INITIAL"},"topState"),pushState:s(function(n){this.begin(n)},"pushState"),stateStackSize:s(function(){return this.conditionStack.length},"stateStackSize"),options:{"case-insensitive":!0},performAction:s(function(n,l,y,d){var c=d;switch(y){case 0:break;case 1:break;case 2:return 10;case 3:break;case 4:break;case 5:return 4;case 6:return 11;case 7:return this.begin("acc_title"),12;break;case 8:return this.popState(),"acc_title_value";break;case 9:return this.begin("acc_descr"),14;break;case 10:return this.popState(),"acc_descr_value";break;case 11:this.begin("acc_descr_multiline");break;case 12:this.popState();break;case 13:return"acc_descr_multiline_value";case 14:return 17;case 15:return 18;case 16:return 19;case 17:return":";case 18:return 6;case 19:return"INVALID"}},"anonymous"),rules:[/^(?:%(?!\{)[^\n]*)/i,/^(?:[^\}]%%[^\n]*)/i,/^(?:[\n]+)/i,/^(?:\s+)/i,/^(?:#[^\n]*)/i,/^(?:journey\b)/i,/^(?:title\s[^#\n;]+)/i,/^(?:accTitle\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*:\s*)/i,/^(?:(?!\n||)*[^\n]*)/i,/^(?:accDescr\s*\{\s*)/i,/^(?:[\}])/i,/^(?:[^\}]*)/i,/^(?:section\s[^#:\n;]+)/i,/^(?:[^#:\n;]+)/i,/^(?::[^#\n;]+)/i,/^(?::)/i,/^(?:$)/i,/^(?:.)/i],conditions:{acc_descr_multiline:{rules:[12,13],inclusive:!1},acc_descr:{rules:[10],inclusive:!1},acc_title:{rules:[8],inclusive:!1},INITIAL:{rules:[0,1,2,3,4,5,6,7,9,11,14,15,16,17,18,19],inclusive:!0}}};return h})();g.lexer=m;function x(){this.yy={}}return s(x,"Parser"),x.prototype=g,g.Parser=x,new x})();K.parser=K;var Pt=K,R="",D=[],L=[],B=[],It=s(function(){D.length=0,L.length=0,R="",B.length=0,lt()},"clear"),At=s(function(t){R=t,D.push(t)},"addSection"),Ft=s(function(){return D},"getSections"),Vt=s(function(){let t=xt(),e=100,a=0;for(;!t&&a<e;)t=xt(),a++;return L.push(...B),L},"getTasks"),Rt=s(function(){let t=[];return L.forEach(a=>{a.people&&t.push(...a.people)}),[...new Set(t)].sort()},"updateActors"),Lt=s(function(t,e){let a=e.substr(1).split(":"),f=0,r=[];a.length===1?(f=Number(a[0]),r=[]):(f=Number(a[0]),r=a[1].split(","));let u=r.map(o=>o.trim()),p={section:R,type:R,people:u,task:t,score:f};B.push(p)},"addTask"),Bt=s(function(t){let e={section:R,type:R,description:t,task:t,classes:[]};L.push(e)},"addTaskOrg"),xt=s(function(){let t=s(function(a){return B[a].processed},"compileTask"),e=!0;for(let[a,f]of B.entries())t(a),e=e&&f.processed;return e},"compileTasks"),jt=s(function(){return Rt()},"getActors"),kt={getConfig:s(()=>F().journey,"getConfig"),clear:It,setDiagramTitle:yt,getDiagramTitle:dt,setAccTitle:ot,getAccTitle:ct,setAccDescription:ht,getAccDescription:ut,addSection:At,getSections:Ft,getTasks:Vt,addTask:Lt,addTaskOrg:Bt,getActors:jt},Nt=s(t=>`.label {
    font-family: ${t.fontFamily};
    color: ${t.textColor};
  }
  .mouth {
    stroke: #666;
  }

  line {
    stroke: ${t.textColor}
  }

  .legend {
    fill: ${t.textColor};
    font-family: ${t.fontFamily};
  }

  .label text {
    fill: #333;
  }
  .label {
    color: ${t.textColor}
  }

  .face {
    ${t.faceColor?`fill: ${t.faceColor}`:"fill: #FFF8DC"};
    stroke: #999;
  }

  .node rect,
  .node circle,
  .node ellipse,
  .node polygon,
  .node path {
    fill: ${t.mainBkg};
    stroke: ${t.nodeBorder};
    stroke-width: 1px;
  }

  .node .label {
    text-align: center;
  }
  .node.clickable {
    cursor: pointer;
  }

  .arrowheadPath {
    fill: ${t.arrowheadColor};
  }

  .edgePath .path {
    stroke: ${t.lineColor};
    stroke-width: 1.5px;
  }

  .flowchart-link {
    stroke: ${t.lineColor};
    fill: none;
  }

  .edgeLabel {
    background-color: ${t.edgeLabelBackground};
    rect {
      opacity: 0.5;
    }
    text-align: center;
  }

  .cluster rect {
  }

  .cluster text {
    fill: ${t.titleColor};
  }

  div.mermaidTooltip {
    position: absolute;
    text-align: center;
    max-width: 200px;
    padding: 2px;
    font-family: ${t.fontFamily};
    font-size: 12px;
    background: ${t.tertiaryColor};
    border: 1px solid ${t.border2};
    border-radius: 2px;
    pointer-events: none;
    z-index: 100;
  }

  .task-type-0, .section-type-0  {
    ${t.fillType0?`fill: ${t.fillType0}`:""};
  }
  .task-type-1, .section-type-1  {
    ${t.fillType0?`fill: ${t.fillType1}`:""};
  }
  .task-type-2, .section-type-2  {
    ${t.fillType0?`fill: ${t.fillType2}`:""};
  }
  .task-type-3, .section-type-3  {
    ${t.fillType0?`fill: ${t.fillType3}`:""};
  }
  .task-type-4, .section-type-4  {
    ${t.fillType0?`fill: ${t.fillType4}`:""};
  }
  .task-type-5, .section-type-5  {
    ${t.fillType0?`fill: ${t.fillType5}`:""};
  }
  .task-type-6, .section-type-6  {
    ${t.fillType0?`fill: ${t.fillType6}`:""};
  }
  .task-type-7, .section-type-7  {
    ${t.fillType0?`fill: ${t.fillType7}`:""};
  }

  .actor-0 {
    ${t.actor0?`fill: ${t.actor0}`:""};
  }
  .actor-1 {
    ${t.actor1?`fill: ${t.actor1}`:""};
  }
  .actor-2 {
    ${t.actor2?`fill: ${t.actor2}`:""};
  }
  .actor-3 {
    ${t.actor3?`fill: ${t.actor3}`:""};
  }
  .actor-4 {
    ${t.actor4?`fill: ${t.actor4}`:""};
  }
  .actor-5 {
    ${t.actor5?`fill: ${t.actor5}`:""};
  }
  ${mt()}
`,"getStyles"),zt=Nt,tt=s(function(t,e){return ft(t,e)},"drawRect"),Wt=s(function(t,e){let f=t.append("circle").attr("cx",e.cx).attr("cy",e.cy).attr("class","face").attr("r",15).attr("stroke-width",2).attr("overflow","visible"),r=t.append("g");r.append("circle").attr("cx",e.cx-15/3).attr("cy",e.cy-15/3).attr("r",1.5).attr("stroke-width",2).attr("fill","#666").attr("stroke","#666"),r.append("circle").attr("cx",e.cx+15/3).attr("cy",e.cy-15/3).attr("r",1.5).attr("stroke-width",2).attr("fill","#666").attr("stroke","#666");function u(g){let m=U().startAngle(Math.PI/2).endAngle(3*(Math.PI/2)).innerRadius(7.5).outerRadius(6.8181818181818175);g.append("path").attr("class","mouth").attr("d",m).attr("transform","translate("+e.cx+","+(e.cy+2)+")")}s(u,"smile");function p(g){let m=U().startAngle(3*Math.PI/2).endAngle(5*(Math.PI/2)).innerRadius(7.5).outerRadius(6.8181818181818175);g.append("path").attr("class","mouth").attr("d",m).attr("transform","translate("+e.cx+","+(e.cy+7)+")")}s(p,"sad");function o(g){g.append("line").attr("class","mouth").attr("stroke",2).attr("x1",e.cx-5).attr("y1",e.cy+7).attr("x2",e.cx+5).attr("y2",e.cy+7).attr("class","mouth").attr("stroke-width","1px").attr("stroke","#666")}return s(o,"ambivalent"),e.score>3?u(r):e.score<3?p(r):o(r),f},"drawFace"),vt=s(function(t,e){let a=t.append("circle");return a.attr("cx",e.cx),a.attr("cy",e.cy),a.attr("class","actor-"+e.pos),a.attr("fill",e.fill),a.attr("stroke",e.stroke),a.attr("r",e.r),a.class!==void 0&&a.attr("class",a.class),e.title!==void 0&&a.append("title").text(e.title),a},"drawCircle"),wt=s(function(t,e){return gt(t,e)},"drawText"),Ot=s(function(t,e){function a(r,u,p,o,g){return r+","+u+" "+(r+p)+","+u+" "+(r+p)+","+(u+o-g)+" "+(r+p-g*1.2)+","+(u+o)+" "+r+","+(u+o)}s(a,"genPoints");let f=t.append("polygon");f.attr("points",a(e.x,e.y,50,20,7)),f.attr("class","labelBox"),e.y=e.y+e.labelMargin,e.x=e.x+.5*e.labelMargin,wt(t,e)},"drawLabel"),Yt=s(function(t,e,a){let f=t.append("g"),r=Z();r.x=e.x,r.y=e.y,r.fill=e.fill,r.width=a.width*e.taskCount+a.diagramMarginX*(e.taskCount-1),r.height=a.height,r.class="journey-section section-type-"+e.num,r.rx=3,r.ry=3,tt(f,r),Tt(a)(e.text,f,r.x,r.y,r.width,r.height,{class:"journey-section section-type-"+e.num},a,e.colour)},"drawSection"),Q=-1,qt=s(function(t,e,a,f){let r=e.x+a.width/2,u=t.append("g");Q++,u.append("line").attr("id",f+"-task"+Q).attr("x1",r).attr("y1",e.y).attr("x2",r).attr("y2",450).attr("class","task-line").attr("stroke-width","1px").attr("stroke-dasharray","4 2").attr("stroke","#666"),Wt(u,{cx:r,cy:300+(5-e.score)*30,score:e.score});let o=Z();o.x=e.x,o.y=e.y,o.fill=e.fill,o.width=a.width,o.height=a.height,o.class="task task-type-"+e.num,o.rx=3,o.ry=3,tt(u,o);let g=e.x+14;e.people.forEach(m=>{let x=e.actors[m].color,h={cx:g,cy:e.y,r:7,fill:x,stroke:"#000",title:m,pos:e.actors[m].position};vt(u,h),g+=10}),Tt(a)(e.task,u,o.x,o.y,o.width,o.height,{class:"task"},a,e.colour)},"drawTask"),Ht=s(function(t,e){pt(t,e)},"drawBackgroundRect"),Tt=(function(){function t(r,u,p,o,g,m,x,h){let i=u.append("text").attr("x",p+g/2).attr("y",o+m/2+5).style("font-color",h).style("text-anchor","middle").text(r);f(i,x)}s(t,"byText");function e(r,u,p,o,g,m,x,h,i){let{taskFontSize:n,taskFontFamily:l}=h,y=r.split(/<br\s*\/?>/gi);for(let d=0;d<y.length;d++){let c=d*n-n*(y.length-1)/2,_=u.append("text").attr("x",p+g/2).attr("y",o).attr("fill",i).style("text-anchor","middle").style("font-size",n).style("font-family",l);_.append("tspan").attr("x",p+g/2).attr("dy",c).text(y[d]),_.attr("y",o+m/2).attr("dominant-baseline","central").attr("alignment-baseline","central"),f(_,x)}}s(e,"byTspan");function a(r,u,p,o,g,m,x,h){let i=u.append("switch"),l=i.append("foreignObject").attr("x",p).attr("y",o).attr("width",g).attr("height",m).attr("position","fixed").append("xhtml:div").style("display","table").style("height","100%").style("width","100%");l.append("div").attr("class","label").style("display","table-cell").style("text-align","center").style("vertical-align","middle").text(r),e(r,i,p,o,g,m,x,h),f(l,x)}s(a,"byFo");function f(r,u){for(let p in u)p in u&&r.attr(p,u[p])}return s(f,"_setTextAttrs"),function(r){return r.textPlacement==="fo"?a:r.textPlacement==="old"?t:e}})(),Xt=s(function(t,e){Q=-1,t.append("defs").append("marker").attr("id",e+"-arrowhead").attr("refX",5).attr("refY",2).attr("markerWidth",6).attr("markerHeight",4).attr("orient","auto").append("path").attr("d","M 0,0 V 4 L6,2 Z")},"initGraphics"),j={drawRect:tt,drawCircle:vt,drawSection:Yt,drawText:wt,drawLabel:Ot,drawTask:qt,drawBackgroundRect:Ht,initGraphics:Xt},Gt=s(function(t){Object.keys(t).forEach(function(a){$[a]=t[a]})},"setConf"),E={},O=0;function St(t){let e=F().journey,a=e.maxLabelWidth;O=0;let f=60;Object.keys(E).forEach(r=>{let u=E[r].color,p={cx:20,cy:f,r:7,fill:u,stroke:"#000",pos:E[r].position};j.drawCircle(t,p);let o=t.append("text").attr("visibility","hidden").text(r),g=o.node().getBoundingClientRect().width;o.remove();let m=[];if(g<=a)m=[r];else{let x=r.split(" "),h="";o=t.append("text").attr("visibility","hidden"),x.forEach(i=>{let n=h?`${h} ${i}`:i;if(o.text(n),o.node().getBoundingClientRect().width>a){if(h&&m.push(h),h=i,o.text(i),o.node().getBoundingClientRect().width>a){let y="";for(let d of i)y+=d,o.text(y+"-"),o.node().getBoundingClientRect().width>a&&(m.push(y.slice(0,-1)+"-"),y=d);h=y}}else h=n}),h&&m.push(h),o.remove()}m.forEach((x,h)=>{let i={x:40,y:f+7+h*20,fill:"#666",text:x,textMargin:e.boxTextMargin??5},l=j.drawText(t,i).node().getBoundingClientRect().width;l>O&&l>e.leftMargin-l&&(O=l)}),f+=Math.max(20,m.length*20)})}s(St,"drawActorLegend");var $=F().journey,P=0,Ut=s(function(t,e,a,f){let r=F(),u=r.journey.titleColor,p=r.journey.titleFontSize,o=r.journey.titleFontFamily,g=r.securityLevel,m;g==="sandbox"&&(m=W("#i"+e));let x=g==="sandbox"?W(m.nodes()[0].contentDocument.body):W("body");S.init();let h=x.select("#"+e);j.initGraphics(h,e);let i=f.db.getTasks(),n=f.db.getDiagramTitle(),l=f.db.getActors();for(let C in E)delete E[C];let y=0;l.forEach(C=>{E[C]={color:$.actorColours[y%$.actorColours.length],position:y},y++}),St(h),P=$.leftMargin+O,S.insert(0,0,P,Object.keys(E).length*50),Zt(h,i,0,e);let d=S.getBounds();n&&h.append("text").text(n).attr("x",P).attr("font-size",p).attr("font-weight","bold").attr("y",25).attr("fill",u).attr("font-family",o);let c=d.stopy-d.starty+2*$.diagramMarginY,_=P+d.stopx+2*$.diagramMarginX;at(h,c,_,$.useMaxWidth),h.append("line").attr("x1",P).attr("y1",$.height*4).attr("x2",_-P-4).attr("y2",$.height*4).attr("stroke-width",4).attr("stroke","black").attr("marker-end","url(#"+e+"-arrowhead)");let k=n?70:0;h.attr("viewBox",`${d.startx} -25 ${_} ${c+k}`),h.attr("preserveAspectRatio","xMinYMin meet"),h.attr("height",c+k+25)},"draw"),S={data:{startx:void 0,stopx:void 0,starty:void 0,stopy:void 0},verticalPos:0,sequenceItems:[],init:s(function(){this.sequenceItems=[],this.data={startx:void 0,stopx:void 0,starty:void 0,stopy:void 0},this.verticalPos=0},"init"),updateVal:s(function(t,e,a,f){t[e]===void 0?t[e]=a:t[e]=f(a,t[e])},"updateVal"),updateBounds:s(function(t,e,a,f){let r=F().journey,u=this,p=0;function o(g){return s(function(x){p++;let h=u.sequenceItems.length-p+1;u.updateVal(x,"starty",e-h*r.boxMargin,Math.min),u.updateVal(x,"stopy",f+h*r.boxMargin,Math.max),u.updateVal(S.data,"startx",t-h*r.boxMargin,Math.min),u.updateVal(S.data,"stopx",a+h*r.boxMargin,Math.max),g!=="activation"&&(u.updateVal(x,"startx",t-h*r.boxMargin,Math.min),u.updateVal(x,"stopx",a+h*r.boxMargin,Math.max),u.updateVal(S.data,"starty",e-h*r.boxMargin,Math.min),u.updateVal(S.data,"stopy",f+h*r.boxMargin,Math.max))},"updateItemBounds")}s(o,"updateFn"),this.sequenceItems.forEach(o())},"updateBounds"),insert:s(function(t,e,a,f){let r=Math.min(t,a),u=Math.max(t,a),p=Math.min(e,f),o=Math.max(e,f);this.updateVal(S.data,"startx",r,Math.min),this.updateVal(S.data,"starty",p,Math.min),this.updateVal(S.data,"stopx",u,Math.max),this.updateVal(S.data,"stopy",o,Math.max),this.updateBounds(r,p,u,o)},"insert"),bumpVerticalPos:s(function(t){this.verticalPos=this.verticalPos+t,this.data.stopy=this.verticalPos},"bumpVerticalPos"),getVerticalPos:s(function(){return this.verticalPos},"getVerticalPos"),getBounds:s(function(){return this.data},"getBounds")},J=$.sectionFills,bt=$.sectionColours,Zt=s(function(t,e,a,f){let r=F().journey,u="",p=r.height*2+r.diagramMarginY,o=a+p,g=0,m="#CCC",x="black",h=0;for(let[i,n]of e.entries()){if(u!==n.section){m=J[g%J.length],h=g%J.length,x=bt[g%bt.length];let y=0,d=n.section;for(let _=i;_<e.length&&e[_].section==d;_++)y=y+1;let c={x:i*r.taskMargin+i*r.width+P,y:50,text:n.section,fill:m,num:h,colour:x,taskCount:y};j.drawSection(t,c,r),u=n.section,g++}let l=n.people.reduce((y,d)=>(E[d]&&(y[d]=E[d]),y),{});n.x=i*r.taskMargin+i*r.width+P,n.y=o,n.width=r.diagramMarginX,n.height=r.diagramMarginY,n.colour=x,n.fill=m,n.num=h,n.actors=l,j.drawTask(t,n,r,f),S.insert(n.x,n.y,n.x+n.width+r.taskMargin,450)}},"drawTasks"),_t={setConf:Gt,draw:Ut},ie={parser:Pt,db:kt,renderer:_t,styles:zt,init:s(t=>{_t.setConf(t.journey),kt.clear()},"init")};export{ie as diagram};
