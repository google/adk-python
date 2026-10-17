import{a as X}from"./chunk-IDCXNWW3.js";import{H as Y,O as P,T as L,_ as I,i as B,u as N}from"./chunk-BTIP5KPZ.js";import{g as x}from"./chunk-JHJZJ2D5.js";import{a as p}from"./chunk-YUSHYV7C.js";import{a as T,b as v,c as _}from"./chunk-BZGM3F3K.js";var A="",M="",O="",D=[],z=new Map,S=p(e=>Y(e,I()),"sanitizeText"),R=p(e=>{switch(e.type){case"terminal":return v(T({},e),{value:S(e.value)});case"nonterminal":return v(T({},e),{name:S(e.name)});case"sequence":return v(T({},e),{elements:e.elements.map(R)});case"choice":return v(T({},e),{alternatives:e.alternatives.map(R)});case"optional":return v(T({},e),{element:R(e.element)});case"repetition":return v(T({},e),{element:R(e.element),separator:e.separator?R(e.separator):void 0});case"special":return v(T({},e),{text:S(e.text)})}},"sanitizeAstNode"),J=p(()=>{A="",M="",O="",D.length=0,z.clear(),L(),x.debug("[Railroad] Database cleared")},"clear"),q=p(e=>{A=S(e),x.debug("[Railroad] Title set:",e)},"setTitle"),G=p(()=>A,"getTitle"),Q=p(e=>{let i=v(T({},e),{name:S(e.name),definition:R(e.definition),comment:e.comment?S(e.comment):void 0});x.debug("[Railroad] Adding rule:",i.name),z.has(i.name)&&x.warn(`[Railroad] Rule '${i.name}' is already defined. Overwriting.`),D.push(i),z.set(i.name,i)},"addRule"),Z=p(()=>D,"getRules"),V=p(e=>z.get(e),"getRule"),ee=p(e=>{M=S(e).replace(/^\s+/g,""),x.debug("[Railroad] Accessibility title set:",e)},"setAccTitle"),te=p(()=>M,"getAccTitle"),re=p(e=>{O=S(e).replace(/\n\s+/g,`
`),x.debug("[Railroad] Accessibility description set:",e)},"setAccDescription"),ie=p(()=>O,"getAccDescription"),ne=q,ae=G,oe={clear:J,setTitle:q,getTitle:G,addRule:Q,getRules:Z,getRule:V,setAccTitle:ee,getAccTitle:te,setAccDescription:re,getAccDescription:ie,setDiagramTitle:ne,getDiagramTitle:ae},g={compactMode:!1,padding:10,verticalSeparation:8,horizontalSeparation:10,arcRadius:10,fontSize:14,fontFamily:"monospace",terminalFill:"#FFFFC0",terminalStroke:"#000000",terminalTextColor:"#000000",nonTerminalFill:"#FFFFFF",nonTerminalStroke:"#000000",nonTerminalTextColor:"#000000",lineColor:"#000000",strokeWidth:2,markerFill:"#000000",commentFill:"#E8E8E8",commentStroke:"#888888",commentTextColor:"#666666",specialFill:"#F0E0FF",specialStroke:"#8800CC",ruleNameColor:"#000066",showMarkers:!0,markerRadius:5},le=/^#(?:[\da-f]{3,4}|[\da-f]{6}|[\da-f]{8})$|^(?:rgb|rgba|hsl|hsla|hwb|lab|lch|oklab|oklch)\([\d\s%+,./-]+\)$|^[a-z]+$/i,se=/^[\w "',.-]+$/,de=new Set(["compactMode","padding","verticalSeparation","horizontalSeparation","arcRadius","fontSize","fontFamily","terminalFill","terminalStroke","terminalTextColor","nonTerminalFill","nonTerminalStroke","nonTerminalTextColor","lineColor","strokeWidth","markerFill","commentFill","commentStroke","commentTextColor","specialFill","specialStroke","ruleNameColor","showMarkers","markerRadius"]),U=p(e=>e?Object.keys(e).every(i=>i==="railroad"||de.has(i)):!1,"isRailroadStyleOptions"),ce=p(e=>e?"railroad"in e&&e.railroad?e.railroad:U(e)?e:{}:{},"extractRailroadOverrides"),me=p(e=>{if(!e||U(e))return{};let d=e,{railroad:i,svgId:n,theme:r,look:t}=d;return _(d,["railroad","svgId","theme","look"])},"extractThemeOverrides"),h=p((e,i)=>{if(typeof e!="string")return i;let n=e.trim();return le.test(n)?n:i},"sanitizeColorValue"),j=p((e,i)=>{if(typeof e!="string")return i;let n=e.trim();return se.test(n)?n:i},"sanitizeFontFamilyValue"),y=p((e,i)=>{let n=typeof e=="number"?e:typeof e=="string"?Number.parseFloat(e):Number.NaN;return Number.isFinite(n)&&n>=0?n:i},"sanitizeNumberValue"),he=p(e=>{let i=typeof e=="number"?e:typeof e=="string"?Number.parseFloat(e):Number.NaN;return Number.isFinite(i)&&i>0?i:void 0},"parseThemeFontSize"),pe=p(e=>{let i=j(e.fontFamily,g.fontFamily),n=he(e.fontSize)??g.fontSize;return v(T({},g),{fontFamily:i,fontSize:n,terminalFill:h(e.secondBkg??e.secondaryColor,g.terminalFill),terminalStroke:h(e.secondaryBorderColor??e.lineColor,g.terminalStroke),terminalTextColor:h(e.secondaryTextColor??e.textColor,g.terminalTextColor),nonTerminalFill:h(e.mainBkg??e.background,g.nonTerminalFill),nonTerminalStroke:h(e.primaryBorderColor??e.lineColor,g.nonTerminalStroke),nonTerminalTextColor:h(e.primaryTextColor??e.textColor,g.nonTerminalTextColor),lineColor:h(e.lineColor,g.lineColor),markerFill:h(e.lineColor,g.markerFill),commentFill:h(e.labelBackground??e.tertiaryColor,g.commentFill),commentStroke:h(e.tertiaryBorderColor??e.lineColor,g.commentStroke),commentTextColor:h(e.tertiaryTextColor??e.textColor,g.commentTextColor),specialFill:h(e.tertiaryColor??e.secondaryColor,g.specialFill),specialStroke:h(e.tertiaryBorderColor??e.secondaryBorderColor,g.specialStroke),ruleNameColor:h(e.titleColor??e.textColor,g.ruleNameColor)})},"buildThemeDefaults"),E=p(e=>{let i=N(),n=T(T(T({},B()),i.themeVariables??{}),me(e)),r=pe(n),t=T(T({},i.railroad??{}),ce(e));return{compactMode:t.compactMode??r.compactMode,padding:y(t.padding,r.padding),verticalSeparation:y(t.verticalSeparation,r.verticalSeparation),horizontalSeparation:y(t.horizontalSeparation,r.horizontalSeparation),arcRadius:y(t.arcRadius,r.arcRadius),fontSize:y(t.fontSize,r.fontSize),fontFamily:j(t.fontFamily,r.fontFamily),terminalFill:h(t.terminalFill,r.terminalFill),terminalStroke:h(t.terminalStroke,r.terminalStroke),terminalTextColor:h(t.terminalTextColor,r.terminalTextColor),nonTerminalFill:h(t.nonTerminalFill,r.nonTerminalFill),nonTerminalStroke:h(t.nonTerminalStroke,r.nonTerminalStroke),nonTerminalTextColor:h(t.nonTerminalTextColor,r.nonTerminalTextColor),lineColor:h(t.lineColor,r.lineColor),strokeWidth:y(t.strokeWidth,r.strokeWidth),markerFill:h(t.markerFill,r.markerFill),commentFill:h(t.commentFill,r.commentFill),commentStroke:h(t.commentStroke,r.commentStroke),commentTextColor:h(t.commentTextColor,r.commentTextColor),specialFill:h(t.specialFill,r.specialFill),specialStroke:h(t.specialStroke,r.specialStroke),ruleNameColor:h(t.ruleNameColor,r.ruleNameColor),showMarkers:t.showMarkers??r.showMarkers,markerRadius:y(t.markerRadius,r.markerRadius)}},"buildRailroadStyleOptions"),Ce=p(e=>{let{fontFamily:i,fontSize:n,terminalFill:r,terminalStroke:t,terminalTextColor:a,nonTerminalFill:d,nonTerminalStroke:s,nonTerminalTextColor:o,lineColor:u,strokeWidth:m,markerFill:c,commentFill:C,commentStroke:l,commentTextColor:f,specialFill:$,specialStroke:F,ruleNameColor:k}=E(e);return`
  .railroad-diagram {
    font-family: ${i};
    font-size: ${n}px;
  }

  .railroad-terminal rect {
    fill: ${r};
    stroke: ${t};
    stroke-width: ${m}px;
  }

  .railroad-terminal text {
    fill: ${a};
    font-family: ${i};
    font-size: ${n}px;
    text-anchor: middle;
    dominant-baseline: middle;
  }

  .railroad-nonterminal rect {
    fill: ${d};
    stroke: ${s};
    stroke-width: ${m}px;
  }

  .railroad-nonterminal text {
    fill: ${o};
    font-family: ${i};
    font-size: ${n}px;
    text-anchor: middle;
    dominant-baseline: middle;
  }

  .railroad-line {
    stroke: ${u};
    stroke-width: ${m}px;
    fill: none;
  }

  .railroad-start circle,
  .railroad-end circle {
    fill: ${c};
  }

  .railroad-comment ellipse {
    fill: ${C};
    stroke: ${l};
    stroke-width: ${m}px;
  }

  .railroad-comment text {
    fill: ${f};
    font-style: italic;
    font-family: ${i};
    font-size: ${n}px;
    text-anchor: middle;
    dominant-baseline: middle;
  }

  .railroad-special rect {
    fill: ${$};
    stroke: ${F};
    stroke-width: ${m}px;
    stroke-dasharray: 5,3;
  }

  .railroad-special text {
    fill: ${o};
    font-family: ${i};
    font-size: ${n}px;
    text-anchor: middle;
    dominant-baseline: middle;
  }

  .railroad-rule-name {
    font-weight: bold;
    fill: ${k};
    font-family: ${i};
    font-size: ${n}px;
  }

  .railroad-group {
    /* Grouping container, no specific styles */
  }
`},"getStyles"),w=class{constructor(){this.d=""}static{p(this,"PathBuilder")}moveTo(e,i){return this.d+=`M ${e} ${i} `,this}lineTo(e,i){return this.d+=`L ${e} ${i} `,this}horizontalTo(e){return this.d+=`H ${e} `,this}verticalTo(e){return this.d+=`V ${e} `,this}arcTo(e,i,n,r,t,a,d){return this.d+=`A ${e} ${i} ${n} ${r?1:0} ${t?1:0} ${a} ${d} `,this}build(){return this.d.trim()}},ue=class{constructor(e,i=E()){this.textCache=new Map,this.svg=e,this.config=i}static{p(this,"RailroadRenderer")}measureText(e){if(this.textCache.has(e))return this.textCache.get(e);let i=this.svg.append("text").attr("font-family",this.config.fontFamily).attr("font-size",this.config.fontSize).text(e),n=i.node().getBBox(),r={width:n.width,height:n.height};return i.remove(),this.textCache.set(e,r),r}renderTerminal(e,i){let n=this.measureText(i),r=n.width+this.config.padding*2,t=n.height+this.config.padding*2,a=e.append("g").attr("class","railroad-terminal");return a.append("rect").attr("x",0).attr("y",0).attr("width",r).attr("height",t).attr("rx",10).attr("ry",10),a.append("text").attr("x",r/2).attr("y",t/2).text(i),{element:a.node(),dimensions:{width:r,height:t,up:t/2,down:t/2}}}renderNonTerminal(e,i){let n=this.measureText(i),r=n.width+this.config.padding*2,t=n.height+this.config.padding*2,a=e.append("g").attr("class","railroad-nonterminal");return a.append("rect").attr("x",0).attr("y",0).attr("width",r).attr("height",t),a.append("text").attr("x",r/2).attr("y",t/2).text(i),{element:a.node(),dimensions:{width:r,height:t,up:t/2,down:t/2}}}renderSequence(e,i){let n=i.map(o=>this.renderExpression(e,o)),r=0,t=0,a=0;for(let o of n)r+=o.dimensions.width,t=Math.max(t,o.dimensions.up),a=Math.max(a,o.dimensions.down);r+=(n.length-1)*this.config.horizontalSeparation;let d=e.append("g").attr("class","railroad-sequence"),s=0;for(let o=0;o<n.length;o++){let u=n[o],m=t-u.dimensions.up;if(d.node().appendChild(u.element).setAttribute("transform",`translate(${s}, ${m})`),o<n.length-1){let C=s+u.dimensions.width,l=C+this.config.horizontalSeparation,f=t;d.append("path").attr("class","railroad-line").attr("d",new w().moveTo(C,f).lineTo(l,f).build())}s+=u.dimensions.width+this.config.horizontalSeparation}return{element:d.node(),dimensions:{width:r,height:t+a,up:t,down:a}}}renderChoice(e,i){let n=i.map(c=>this.renderExpression(e,c)),r=0,t=0;for(let c of n)r=Math.max(r,c.dimensions.width),t+=c.dimensions.height;t+=(n.length-1)*this.config.verticalSeparation;let a=this.config.arcRadius,d=a*4,s=r+d,o=e.append("g").attr("class","railroad-choice"),u=0,m=t/2;for(let c of n){let C=u,l=C+c.dimensions.up,f=a*2+(r-c.dimensions.width)/2;o.node().appendChild(c.element).setAttribute("transform",`translate(${f}, ${C})`);let F=new w,k=l>m;l===m?F.moveTo(0,m).lineTo(f,l):F.moveTo(0,m).arcTo(a,a,0,!1,k,a,m+(k?a:-a)).lineTo(a,l-(k?a:-a)).arcTo(a,a,0,!1,!k,a*2,l).lineTo(f,l),o.append("path").attr("class","railroad-line").attr("d",F.build());let b=new w,W=f+c.dimensions.width,K=s-a*2;l===m?b.moveTo(W,l).lineTo(s,m):b.moveTo(W,l).lineTo(K,l).arcTo(a,a,0,!1,!k,s-a,l+(k?-a:a)).lineTo(s-a,m+(k?a:-a)).arcTo(a,a,0,!1,k,s,m),o.append("path").attr("class","railroad-line").attr("d",b.build()),u+=c.dimensions.height+this.config.verticalSeparation}return{element:o.node(),dimensions:{width:s,height:t,up:m,down:t-m}}}renderOptional(e,i){let n=this.renderExpression(e,i),r=this.config.arcRadius,t=r*2,a=n.dimensions.width+r*4,d=n.dimensions.height+t,s=e.append("g").attr("class","railroad-optional"),o=r*2,u=t;s.node().appendChild(n.element).setAttribute("transform",`translate(${o}, ${u})`);let c=u+n.dimensions.up,C=new w().moveTo(0,c).lineTo(r*2,c);s.append("path").attr("class","railroad-line").attr("d",C.build());let l=new w().moveTo(o+n.dimensions.width,c).lineTo(a,c);s.append("path").attr("class","railroad-line").attr("d",l.build());let f=new w().moveTo(0,c).arcTo(r,r,0,!1,!1,r,c-r).lineTo(r,r).arcTo(r,r,0,!1,!0,r*2,0).lineTo(a-r*2,0).arcTo(r,r,0,!1,!0,a-r,r).lineTo(a-r,c-r).arcTo(r,r,0,!1,!1,a,c);return s.append("path").attr("class","railroad-line").attr("d",f.build()),{element:s.node(),dimensions:{width:a,height:d,up:c,down:d-c}}}renderRepetition(e,i,n){let r=this.renderExpression(e,i),t=this.config.arcRadius,a=t*2,d=r.dimensions.width+t*4,s=n===0,o=r.dimensions.height+a+(s?a:0),u=e.append("g").attr("class","railroad-repetition"),m=t*2,c=s?a:0;u.node().appendChild(r.element).setAttribute("transform",`translate(${m}, ${c})`);let l=c+r.dimensions.up;u.append("path").attr("class","railroad-line").attr("d",new w().moveTo(0,l).lineTo(t*2,l).build()),u.append("path").attr("class","railroad-line").attr("d",new w().moveTo(m+r.dimensions.width,l).lineTo(d,l).build());let f=c+r.dimensions.height+t,$=new w().moveTo(m+r.dimensions.width,l).arcTo(t,t,0,!1,!0,m+r.dimensions.width+t,l+t).lineTo(m+r.dimensions.width+t,f).arcTo(t,t,0,!1,!0,m+r.dimensions.width,f+t).lineTo(t*2,f+t).arcTo(t,t,0,!1,!0,t,f).lineTo(t,l+t).arcTo(t,t,0,!1,!0,t*2,l);if(u.append("path").attr("class","railroad-line").attr("d",$.build()),s){let F=new w().moveTo(0,l).arcTo(t,t,0,!1,!1,t,l-t).lineTo(t,t).arcTo(t,t,0,!1,!0,t*2,0).lineTo(d-t*2,0).arcTo(t,t,0,!1,!0,d-t,t).lineTo(d-t,l-t).arcTo(t,t,0,!1,!1,d,l);u.append("path").attr("class","railroad-line").attr("d",F.build())}return{element:u.node(),dimensions:{width:d,height:o,up:l,down:o-l}}}renderSpecial(e,i){let n=this.measureText("? "+i+" ?"),r=n.width+this.config.padding*2,t=n.height+this.config.padding*2,a=e.append("g").attr("class","railroad-special");return a.append("rect").attr("x",0).attr("y",0).attr("width",r).attr("height",t),a.append("text").attr("x",r/2).attr("y",t/2).text("? "+i+" ?"),{element:a.node(),dimensions:{width:r,height:t,up:t/2,down:t/2}}}renderExpression(e,i){switch(i.type){case"terminal":return this.renderTerminal(e,i.value);case"nonterminal":return this.renderNonTerminal(e,i.name);case"sequence":return this.renderSequence(e,i.elements);case"choice":return this.renderChoice(e,i.alternatives);case"optional":return this.renderOptional(e,i.element);case"repetition":return this.renderRepetition(e,i.element,i.min);case"special":return this.renderSpecial(e,i.text);default:throw new Error(`Unknown node type: ${i.type}`)}}renderRule(e,i){let n=this.svg.append("g").attr("class","railroad-rule").attr("transform",`translate(0, ${i})`),r=e.name+" =",t=this.measureText(r).width+20,a=t+20,d=n.append("g"),s=this.renderExpression(d,e.definition),o=Math.max(20,s.dimensions.up),u=o-s.dimensions.up;return d.attr("transform",`translate(${a}, ${u})`),n.append("g").attr("class","railroad-rule-name-group").append("text").attr("class","railroad-rule-name").attr("x",0).attr("y",o).text(r),n.append("g").attr("class","railroad-start").append("circle").attr("cx",t).attr("cy",o).attr("r",this.config.markerRadius),n.append("g").attr("class","railroad-end").append("circle").attr("cx",a+s.dimensions.width+10).attr("cy",o).attr("r",this.config.markerRadius),n.append("path").attr("class","railroad-line").attr("d",new w().moveTo(t+this.config.markerRadius,o).lineTo(a,o).build()),n.append("path").attr("class","railroad-line").attr("d",new w().moveTo(a+s.dimensions.width,o).lineTo(a+s.dimensions.width+10-this.config.markerRadius,o).build()),{height:Math.max(40,u+s.dimensions.height+this.config.padding*2),width:a+s.dimensions.width+10+this.config.markerRadius}}renderDiagram(e){let i=this.config.padding,n=0;for(let r of e){let t=this.renderRule(r,i);i+=t.height+this.config.verticalSeparation,n=Math.max(n,t.width)}return{width:n+this.config.padding*2,height:i+this.config.padding}}},H=p((e,i,n)=>{P(e,i.height,i.width,n),e.attr("viewBox",`0 0 ${i.width} ${i.height}`)},"configureRailroadSvgSize"),ge=p((e,i,n)=>{x.debug(`[Railroad] Rendering diagram
`+e);try{let r=X(i);r.attr("class","railroad-diagram");let a=N().railroad?.useMaxWidth??!0,d=oe.getRules();if(x.debug(`[Railroad] Rendering ${d.length} rules`),d.length===0){x.warn("[Railroad] No rules to render"),H(r,{height:100,width:200},a);return}let o=new ue(r,E()).renderDiagram(d);H(r,o,a),x.debug("[Railroad] Render complete")}catch(r){throw x.error("[Railroad] Render error:",r),r}},"draw"),ke={draw:ge};export{oe as a,Ce as b,ke as c};
