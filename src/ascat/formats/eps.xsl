<?xml version="1.0" encoding="ISO-8859-1"?>
<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

<!-- 
	File: eps.xsl
	Purpose: Stylesheet to transform Eugene EPS format descriptions
                 into xhtml
	$Log: eps.xsl,v $
	Revision 1.1  2003/02/28 10:49:29  melson
	Initial revision


	Copyright (c) Eumetsat 2003, 2004
-->

<xsl:output method="html" />
<xsl:strip-space elements="with-param" />


<xsl:template match="/format">
	<xsl:for-each select="product">
		<xsl:call-template name="productfound">
		</xsl:call-template>
	</xsl:for-each>
</xsl:template>

<xsl:template name="productfound">
	<html><body>
	<h1>Format description for <xsl:value-of select="@name" /></h1>


	<h5>
		<xsl:value-of select="/*/full-description" />
		<xsl:for-each select="/*/version-control">
			(Baseline: <xsl:value-of select="baseline" />)
			(MPHR Format Version: 
			<xsl:for-each select="format-version">
				<xsl:value-of select="major" />.
				<xsl:value-of select="minor" />,
			</xsl:for-each>)
		</xsl:for-each>
	</h5>



	<h2>Contents:</h2>
	<ul>

	<xsl:for-each select="mphr">
		<xsl:call-template name="display-contents">
			<xsl:with-param name="classname">MPHR</xsl:with-param>
			<xsl:with-param name="classnumber">1</xsl:with-param>
			<xsl:with-param name="subclassnumber" select="@subclass" />
			<xsl:with-param name="name" select="@name" />
			<xsl:with-param name="version" select="@version" />
		</xsl:call-template>
	</xsl:for-each>

	<xsl:for-each select="sphr">
		<xsl:call-template name="display-contents">
			<xsl:with-param name="classname">SPHR</xsl:with-param>
			<xsl:with-param name="classnumber">2</xsl:with-param>
			<xsl:with-param name="subclassnumber" select="@subclass" />
			<xsl:with-param name="name" select="@name" />
			<xsl:with-param name="version" select="@version" />
		</xsl:call-template>
	</xsl:for-each>

	<xsl:for-each select="geadr">
		<xsl:call-template name="display-contents">
			<xsl:with-param name="classname">GEADR</xsl:with-param>
			<xsl:with-param name="classnumber">3</xsl:with-param>
			<xsl:with-param name="subclassnumber" select="@subclass" />
			<xsl:with-param name="name" select="@name" />
			<xsl:with-param name="version" select="@version" />
		</xsl:call-template>
	</xsl:for-each>

	<xsl:for-each select="giadr">
		<xsl:call-template name="display-contents">
			<xsl:with-param name="classname">GIADR</xsl:with-param>
			<xsl:with-param name="classnumber">5</xsl:with-param>
			<xsl:with-param name="subclassnumber" select="@subclass" />
			<xsl:with-param name="version" select="@version" />
			<xsl:with-param name="name" select="@name" />
		</xsl:call-template>
	</xsl:for-each>

	<xsl:for-each select="veadr">
		<xsl:call-template name="display-contents">
			<xsl:with-param name="classname">VEADR</xsl:with-param>
			<xsl:with-param name="classnumber">6</xsl:with-param>
			<xsl:with-param name="subclassnumber" select="@subclass" />
			<xsl:with-param name="name" select="@name" />
			<xsl:with-param name="version" select="@version" />
		</xsl:call-template>
	</xsl:for-each>

	<xsl:for-each select="viadr">
		<xsl:call-template name="display-contents">
			<xsl:with-param name="classname">VIADR</xsl:with-param>
			<xsl:with-param name="classnumber">7</xsl:with-param>
			<xsl:with-param name="subclassnumber" select="@subclass" />
			<xsl:with-param name="version" select="@version" />
			<xsl:with-param name="name" select="@name" />
		</xsl:call-template>
	</xsl:for-each>

	<xsl:for-each select="mdr">
		<xsl:call-template name="display-contents">
			<xsl:with-param name="classname">MDR</xsl:with-param>
			<xsl:with-param name="classnumber">8</xsl:with-param>
			<xsl:with-param name="subclassnumber" select="@subclass" />
			<xsl:with-param name="version" select="@version" />
			<xsl:with-param name="name" select="@name" />
		</xsl:call-template>
	</xsl:for-each>
	</ul>


	<xsl:for-each select="mphr">
		<xsl:call-template name="display-toplevel-section">
			<xsl:with-param name="representation">text</xsl:with-param>
			<xsl:with-param name="classname">MPHR</xsl:with-param>
			<xsl:with-param name="classnumber">1</xsl:with-param>
			<xsl:with-param name="subclassnumber" select="@subclass" />
			<xsl:with-param name="version" select="@version" />
			<xsl:with-param name="name" select="@name" />
		</xsl:call-template>
	</xsl:for-each>

	<xsl:for-each select="sphr">
		<xsl:call-template name="display-toplevel-section">
			<xsl:with-param name="representation">text</xsl:with-param>
			<xsl:with-param name="classname">SPHR</xsl:with-param>
			<xsl:with-param name="classnumber">2</xsl:with-param>
			<xsl:with-param name="subclassnumber" select="@subclass" />
			<xsl:with-param name="version" select="@version" />
			<xsl:with-param name="name" select="@name" />
		</xsl:call-template>
	</xsl:for-each>

	<xsl:for-each select="geadr">
		<xsl:call-template name="display-eadr">
			<xsl:with-param name="representation">binary</xsl:with-param>
			<xsl:with-param name="classname">GEADR</xsl:with-param>
			<xsl:with-param name="classnumber">3</xsl:with-param>
			<xsl:with-param name="subclassnumber" select="@subclass" />
			<xsl:with-param name="name" select="@name" />
			<xsl:with-param name="version" select="@version" />
		</xsl:call-template>
	</xsl:for-each>

	<xsl:for-each select="giadr">
		<xsl:call-template name="display-toplevel-section">
			<xsl:with-param name="representation">binary</xsl:with-param>
			<xsl:with-param name="classname">GIADR</xsl:with-param>
			<xsl:with-param name="classnumber">5</xsl:with-param>
			<xsl:with-param name="subclassnumber" select="@subclass" />
			<xsl:with-param name="version" select="@version" />
			<xsl:with-param name="name" select="@name" />
		</xsl:call-template>
	</xsl:for-each>
			
	<xsl:for-each select="veadr">
		<xsl:call-template name="display-eadr">
			<xsl:with-param name="representation">binary</xsl:with-param>
			<xsl:with-param name="classname">VEADR</xsl:with-param>
			<xsl:with-param name="classnumber">6</xsl:with-param>
			<xsl:with-param name="subclassnumber" select="@subclass" />
			<xsl:with-param name="name" select="@name" />
			<xsl:with-param name="version" select="@version" />
		</xsl:call-template>
	</xsl:for-each>

	<xsl:for-each select="viadr">
		<xsl:call-template name="display-toplevel-section">
			<xsl:with-param name="representation">binary</xsl:with-param>
			<xsl:with-param name="classname">VIADR</xsl:with-param>
			<xsl:with-param name="classnumber">7</xsl:with-param>
			<xsl:with-param name="subclassnumber" select="@subclass" />
			<xsl:with-param name="version" select="@version" />
			<xsl:with-param name="name" select="@name" />
		</xsl:call-template>
	</xsl:for-each>

	<xsl:for-each select="mdr">
		<xsl:call-template name="display-toplevel-section">
			<xsl:with-param name="representation">binary</xsl:with-param>
			<xsl:with-param name="classname">MDR</xsl:with-param>
			<xsl:with-param name="classnumber">8</xsl:with-param>
			<xsl:with-param name="subclassnumber" select="@subclass" />
			<xsl:with-param name="version" select="@version" />
			<xsl:with-param name="name" select="@name" />
		</xsl:call-template>
	</xsl:for-each>

	<!-- To handle fields that are of type enumerated -->
	<xsl:for-each select="//field[@type='enumerated']">
		<xsl:sort select="@name" order="ascending"/>
		<xsl:call-template name="doenumeratedtest">
			<xsl:with-param name="pos" select="position()" />
			<xsl:with-param name="name" select="@name" />
			<xsl:with-param name="type">field</xsl:with-param>
		</xsl:call-template>
	</xsl:for-each>

	<!-- To handle arrays and complex structures that have a field of type enumerated -->
	<xsl:for-each select="//array">
		<xsl:sort select="@name" order="ascending"/>
		<xsl:variable name="tmpname" select="@name"/>
		<xsl:variable name="tmppos" select="position()"/>
		<xsl:for-each select="array">
		<xsl:for-each select="field[@type='enumerated']">
			<xsl:choose>
			<xsl:when test="@name">
				<!-- Do nothing -->
			</xsl:when>
			<xsl:otherwise>
				<xsl:call-template name="doenumeratedtest">
					<xsl:with-param name="pos" select="$tmppos" />
					<xsl:with-param name="name" select="$tmpname" />
					<xsl:with-param name="type">array2</xsl:with-param>
				</xsl:call-template>
			</xsl:otherwise>
			</xsl:choose>
		</xsl:for-each>
		</xsl:for-each>
		<xsl:for-each select="field[@type='enumerated']">
			<xsl:choose>
			<xsl:when test="@name">
				<!-- Do nothing -->
			</xsl:when>
			<xsl:otherwise>
				<xsl:call-template name="doenumeratedtest">
					<xsl:with-param name="pos" select="$tmppos" />
					<xsl:with-param name="name" select="$tmpname" />
					<xsl:with-param name="type">array</xsl:with-param>
				</xsl:call-template>
			</xsl:otherwise>
			</xsl:choose>
		</xsl:for-each>
	</xsl:for-each>


	<!-- To handle fields that are of type bitfield -->
	<xsl:for-each select="//field[@type='bitfield']">
		<xsl:sort select="@name" order="ascending"/>
		<xsl:call-template name="dobitfieldtest">
			<xsl:with-param name="bitfieldpos" select="position()" />
			<xsl:with-param name="bitfieldname" select="@name" />
			<xsl:with-param name="bitfieldtype">field</xsl:with-param>
		</xsl:call-template>
	</xsl:for-each>

	<!-- To handle arrays and complex structures that have a field of type bitfield -->
	<xsl:for-each select="//array">
		<xsl:sort select="@name" order="ascending"/>
		<xsl:variable name="tmpname" select="@name"/>
		<xsl:variable name="tmppos" select="position()"/>
		<xsl:for-each select="array">
		<xsl:for-each select="field[@type='bitfield']">
			<xsl:choose>
			<xsl:when test="@name">
				<!-- Do nothing -->
			</xsl:when>
			<xsl:otherwise>
				<xsl:call-template name="dobitfieldtest">
					<xsl:with-param name="bitfieldpos" select="$tmppos" />
					<xsl:with-param name="bitfieldname" select="$tmpname"/>
					<xsl:with-param name="bitfieldtype">array2</xsl:with-param>
				</xsl:call-template>
			</xsl:otherwise>
			</xsl:choose>
		</xsl:for-each>
		</xsl:for-each>
		<xsl:for-each select="field[@type='bitfield']">
			<xsl:choose>
			<xsl:when test="@name">
				<!-- Do nothing -->
			</xsl:when>
			<xsl:otherwise>
				<xsl:call-template name="dobitfieldtest">
					<xsl:with-param name="bitfieldpos" select="$tmppos" />
					<xsl:with-param name="bitfieldname" select="$tmpname"/>
					<xsl:with-param name="bitfieldtype">array</xsl:with-param>
				</xsl:call-template>
			</xsl:otherwise>
			</xsl:choose>
		</xsl:for-each>
	</xsl:for-each>

	<xsl:for-each select="//parameters">
		<xsl:call-template name="display-parameters">
		</xsl:call-template>
	</xsl:for-each>

	</body></html>
</xsl:template>


<xsl:template name="display-toplevel-section">
	<xsl:param name="representation" />
	<xsl:param name="classname"/>
	<xsl:param name="classnumber" />
	<xsl:param name="subclassnumber" />
	<xsl:param name="version" />
	<xsl:param name="name"></xsl:param>

	<xsl:call-template name="display-toplevel-intro">
		<xsl:with-param name="classname" select="$classname" />
		<xsl:with-param name="classnumber" select="$classnumber" />
		<xsl:with-param name="subclassnumber" select="$subclassnumber" />
		<xsl:with-param name="name" select="$name" />
		<xsl:with-param name="version" select="$version" />
	</xsl:call-template>

	<table border="1">

		<xsl:call-template name="display-table-banner" />

	<tbody>
		<xsl:variable name="beginner">
			<xsl:for-each select="*[1]">
				<xsl:value-of select="name()" />
			</xsl:for-each>
		</xsl:variable>

		<!-- Decomment this section to get a grey dividing line after all GRH entries
		<xsl:if test="$beginner != 'delimiter'">
			<tr bgcolor="#a0a0a0" valign="top">
				<td colspan="12">
					<p></p>
				</td>
			</tr>
		</xsl:if>			
		-->

		<xsl:call-template name="display-child">
			<xsl:with-param name="pos" select="1" />
			<xsl:with-param name="offset" select="20" />
			<xsl:with-param name="representation" select="$representation" />
			<xsl:with-param name="Dim1" select="1" />
			<xsl:with-param name="Dim2" select="1" />
			<xsl:with-param name="Dim3" select="1" />
			<xsl:with-param name="Dim4" select="1" />
			<xsl:with-param name="toplevel" select="10" />
		</xsl:call-template>
	</tbody>
	</table>

	<br/>

</xsl:template>

<xsl:template name="display-child">

	<!-- recursive template to walk through a section
		displaying all field elements -->

	<xsl:param name="pos" />
	<xsl:param name="offset" />
	<xsl:param name="representation" />
	<xsl:param name="Dim1" select="1" />
	<xsl:param name="Dim2" select="1" />
	<xsl:param name="Dim3" select="1" />
	<xsl:param name="Dim4" select="1" />
	<xsl:param name="name"></xsl:param>
	<xsl:param name="toplevel" select="0" />
	<xsl:param name="subsection" select="0" />
	<xsl:param name="bgcolour">ffffff</xsl:param>

	<xsl:variable name="class">
		<xsl:for-each select="*[$pos]">
			<xsl:value-of select="name()" />
		</xsl:for-each>
	</xsl:variable>

	<xsl:choose><xsl:when test="count(*) >= $pos">


		<!-- Dim1 -->
		<xsl:variable name="tmpDim1">
			<xsl:choose>
			<xsl:when test="contains($Dim1, '$')">
				<xsl:call-template name="getparamvalue">
					<xsl:with-param name="name" select="$Dim1" />
				</xsl:call-template>
			</xsl:when>
			<xsl:otherwise>
				<xsl:value-of select="$Dim1" />
			</xsl:otherwise>
			</xsl:choose>	
		</xsl:variable>

		<!-- Dim2 -->
		<xsl:variable name="tmpDim2">
			<xsl:choose>
			<xsl:when test="contains($Dim2, '$')">
				<xsl:call-template name="getparamvalue">
					<xsl:with-param name="name" select="$Dim2" />
				</xsl:call-template>
			</xsl:when>
			<xsl:otherwise>
				<xsl:value-of select="$Dim2" />
			</xsl:otherwise>
			</xsl:choose>	
		</xsl:variable>

		<!-- Dim3 -->
		<xsl:variable name="tmpDim3">
			<xsl:choose>
			<xsl:when test="contains($Dim3, '$')">
				<xsl:call-template name="getparamvalue">
					<xsl:with-param name="name" select="$Dim3" />
				</xsl:call-template>
			</xsl:when>
			<xsl:otherwise>
				<xsl:value-of select="$Dim3" />
			</xsl:otherwise>
			</xsl:choose>	
		</xsl:variable>
	
		<!-- Dim4 -->
		<xsl:variable name="tmpDim4">
			<xsl:choose>
			<xsl:when test="contains($Dim4, '$')">
				<xsl:call-template name="getparamvalue">
					<xsl:with-param name="name" select="$Dim4" />
				</xsl:call-template>
			</xsl:when>
			<xsl:otherwise>
				<xsl:value-of select="$Dim4" />
			</xsl:otherwise>
			</xsl:choose>	
		</xsl:variable>


		<!-- what type is the next element? -->

		<xsl:choose>

			<xsl:when test="$class = 'delimiter'">

				<!-- Display the banner -->

				<tr valign="top">
					<td bgcolor="#a0a0a0" colspan="12">
						<xsl:value-of select="*[$pos]/@name" />
					</td>
				</tr>

				<!-- Continue the loop -->

				<xsl:call-template name="display-child">
					<xsl:with-param name="pos" select="$pos + 1" />
					<xsl:with-param name="offset" select="$offset" />
					<xsl:with-param name="representation" select="$representation" />
					<xsl:with-param name="Dim1" select="1" />
					<xsl:with-param name="Dim2" select="1" />
					<xsl:with-param name="Dim3" select="1" />
					<xsl:with-param name="Dim4" select="1" />
					<xsl:with-param name="toplevel" select="$toplevel" />
				</xsl:call-template>
			</xsl:when>


			<xsl:when test="$class = 'if'">
				<!-- First display the section heading -->

				<tr valign="top">
					<td bgcolor="#f0f0a0" colspan="12">
						<xsl:value-of select="concat(concat(concat('If ', *[$pos]/@source), concat(' = ', *[$pos]/@value)), ' then')" />
					</td>
				</tr>

				<!-- Then display the children -->

				<xsl:for-each select="*[$pos]">
					<xsl:call-template name="display-child">
						<xsl:with-param name="pos" select="1" />
						<xsl:with-param name="offset" select="$offset" />
						<xsl:with-param name="representation" select="$representation" />
						<xsl:with-param name="Dim1" select="1" />
						<xsl:with-param name="Dim2" select="1" />
						<xsl:with-param name="Dim3" select="1" />
						<xsl:with-param name="Dim4" select="1" />
						<xsl:with-param name="bgcolour">f0f0c0</xsl:with-param>
					</xsl:call-template>
				</xsl:for-each>

				<tr valign="top">
					<td bgcolor="#f0f0a0" colspan="12">
						<xsl:value-of select="concat('End If', '')" />
					</td>
				</tr>

				<!-- Find how big the childs are -->

				<xsl:variable name="section-childs-length">
					<xsl:for-each select="*[$pos]">
						<xsl:call-template name="recursive-calculate-length">
							<xsl:with-param name="representation" 
								select="$representation" />
						</xsl:call-template>
					</xsl:for-each>
				</xsl:variable>

				<!--  CHILD-LENGTH <xsl:value-of select="$childs-length" /> X -->

				<!-- And continue iterating -->

				<xsl:call-template name="display-child">
					<xsl:with-param name="pos" select="$pos + 1" />
					<xsl:with-param name="offset" select="$offset + $section-childs-length" />
					<xsl:with-param name="representation" select="$representation" />
					<xsl:with-param name="Dim1" select="1" />
					<xsl:with-param name="Dim2" select="1" />
					<xsl:with-param name="Dim3" select="1" />
					<xsl:with-param name="Dim4" select="1" />
					<xsl:with-param name="toplevel" select="$toplevel" />
					<xsl:with-param name="bgcolour" select="$bgcolour" />
				</xsl:call-template>
			</xsl:when>


			<xsl:when test="$class = 'section'">

				<!-- First display the section heading -->

				<tr valign="top">
					<td bgcolor="#f0f0a0" colspan="12">
						<xsl:value-of select="*[$pos]/@name" />
					</td>
				</tr>

				<!-- Then display the children -->

				<xsl:for-each select="*[$pos]">
					<xsl:call-template name="display-child">
						<xsl:with-param name="pos" select="1" />
						<xsl:with-param name="offset" select="$offset" />
						<xsl:with-param name="representation" select="$representation" />
						<xsl:with-param name="Dim1" select="1" />
						<xsl:with-param name="Dim2" select="1" />
						<xsl:with-param name="Dim3" select="1" />
						<xsl:with-param name="Dim4" select="1" />
						<xsl:with-param name="bgcolour">f0f0c0</xsl:with-param>
					</xsl:call-template>
				</xsl:for-each>

				<tr valign="top">
					<td bgcolor="#f0f0a0" colspan="12">
						<xsl:value-of select="concat('End: ', *[$pos]/@name)" />
					</td>
				</tr>

				<!-- Find how big the childs are -->

				<xsl:variable name="section-childs-length">
					<xsl:for-each select="*[$pos]">
						<xsl:call-template name="recursive-calculate-length">
							<xsl:with-param name="representation" 
								select="$representation" />
						</xsl:call-template>
					</xsl:for-each>
				</xsl:variable>

				<!--  CHILD-LENGTH <xsl:value-of select="$childs-length" /> X -->

				<!-- And continue iterating -->

				<xsl:call-template name="display-child">
					<xsl:with-param name="pos" select="$pos + 1" />
					<xsl:with-param name="offset" select="$offset + $section-childs-length" />
					<xsl:with-param name="representation" select="$representation" />
					<xsl:with-param name="Dim1" select="1" />
					<xsl:with-param name="Dim2" select="1" />
					<xsl:with-param name="Dim3" select="1" />
					<xsl:with-param name="Dim4" select="1" />
					<xsl:with-param name="toplevel" select="$toplevel" />
					<xsl:with-param name="bgcolour" select="$bgcolour" />
				</xsl:call-template>

			</xsl:when>

			<xsl:when test="$class = 'array'">

				<!-- Find how big the childs are -->

				<xsl:variable name="childs-length">
					<xsl:for-each select="*[$pos]">
						<xsl:call-template name="recursive-calculate-length">
							<xsl:with-param name="representation" 
								select="$representation" />
						</xsl:call-template>
					</xsl:for-each>
				</xsl:variable>

				<!-- Dont display the contents now, let the recusive call do that -->

				<xsl:for-each select="*[$pos]">	

					<!-- length -->
					<xsl:variable name="tmplength">
						<xsl:choose>
						<xsl:when test="contains(@length, '$')">
							<xsl:call-template name="getparamvalue">
								<xsl:with-param name="name" select="@length" />
							</xsl:call-template>
						</xsl:when>
						<xsl:otherwise>
							<xsl:value-of select="@length" />
						</xsl:otherwise>
						</xsl:choose>	
					</xsl:variable>

					<xsl:choose>

						<xsl:when test="count(*) = 1">
							<xsl:call-template name="display-child">
								<xsl:with-param name="name" 
									select="concat(@name, $name)" />
								<xsl:with-param name="pos" select="1" />
								<xsl:with-param name="offset" select="$offset" />
								<xsl:with-param name="representation" 
									select="$representation" />
								<xsl:with-param name="Dim1" select="@length" />
								<xsl:with-param name="Dim2" select="$Dim1" />
								<xsl:with-param name="Dim3" select="$Dim2" />
								<xsl:with-param name="Dim4" select="$Dim3" />
								<xsl:with-param name="subsection" 
									select="$subsection" />
								<xsl:with-param name="bgcolour" select="$bgcolour" />
							</xsl:call-template>	
						</xsl:when>

						<xsl:otherwise>

							<tr valign="top" bgcolor="{concat('#', $bgcolour)}">
								<xsl:choose>
   									<xsl:when test="@name != ''">
  										<td><xsl:value-of select="@name" /></td>
   									</xsl:when>
									<xsl:otherwise>
     										<td><xsl:value-of select="$name" /></td>
   									</xsl:otherwise>
								</xsl:choose>
								<td><xsl:value-of select="@description" /></td>
								<td><font color="White">.</font></td>
								<td><font color="White">.</font></td>
								<td>
								<xsl:choose>
								<xsl:when test="contains(@length, '$')">
									<a href="{concat('#', substring(@length, 2))}"><xsl:value-of select="substring(@length, 2)" /></a>
								</xsl:when>
								<xsl:otherwise>
									<xsl:value-of select="@length" />
								</xsl:otherwise>
								</xsl:choose>	
								</td>
								<td>
								<xsl:choose>
								<xsl:when test="contains($Dim1, '$')">
									<a href="{concat('#', substring($Dim1, 2))}"><xsl:value-of select="substring($Dim1, 2)" /></a>
								</xsl:when>
								<xsl:otherwise>
									<xsl:value-of select="$Dim1" />
								</xsl:otherwise>
								</xsl:choose>
								</td>
								<td>
								<xsl:choose>
								<xsl:when test="contains($Dim2, '$')">
									<a href="{concat('#', substring($Dim2, 2))}"><xsl:value-of select="substring($Dim2, 2)" /></a>
								</xsl:when>
								<xsl:otherwise>
									<xsl:value-of select="$Dim2" />
								</xsl:otherwise>
								</xsl:choose>
								</td>
								<td>
								<xsl:choose>
								<xsl:when test="contains($Dim3, '$')">
									<a href="{concat('#', substring($Dim3, 2))}"><xsl:value-of select="substring($Dim3, 2)" /></a>
								</xsl:when>
								<xsl:otherwise>
									<xsl:value-of select="$Dim3" />
								</xsl:otherwise>
								</xsl:choose>
								</td>
								<td><xsl:value-of select="@type" /></td>
								<td><xsl:value-of select="$childs-length"/></td>
								<td><xsl:value-of select="$childs-length * 
									$tmpDim1 * $tmpDim2 * $tmpDim3 * $tmpDim4 * $tmplength"/></td>
								<td><xsl:value-of select="$offset" /></td>
							</tr>
							
							<xsl:call-template name="display-child">
								<xsl:with-param name="pos" select="1" />
								<xsl:with-param name="subsection" select="1" />
								<xsl:with-param name="representation"
									select="$representation" />
								<xsl:with-param name="bgcolour" select="$bgcolour" />
							</xsl:call-template>
							
						</xsl:otherwise>

					</xsl:choose>

				</xsl:for-each>

				<!-- Find how big the childs are -->

				<xsl:variable name="array-childs-length">
					<xsl:for-each select="*[$pos]">
						<xsl:call-template name="recursive-calculate-length">
							<xsl:with-param name="representation" 
								select="$representation" />
						</xsl:call-template>
					</xsl:for-each>
				</xsl:variable>

				<!-- Continue iterating at this level -->

				<!-- ATLENGTH <xsl:value-of select="*[$pos]/@length" /> HTGNELTA -->

				<!--
				Dim1 <xsl:value-of select="$Dim1"/> Dim2 <xsl:value-of select="$Dim2"/>
				Dim3 <xsl:value-of select="$Dim3"/> Dim4 <xsl:value-of select="$Dim4"/> 
				CHILDLENGTH <xsl:value-of select="$array-childs-length"/>
				LENGTH <xsl:value-of select="*[$pos]/@length"/>
				-->

				<!-- length -->
				<xsl:variable name="testlength">
					<xsl:choose>
					<xsl:when test="contains(*[$pos]/@length, '$')">
						<xsl:call-template name="getparamvalue">
							<xsl:with-param name="name" select="*[$pos]/@length" />
						</xsl:call-template>
					</xsl:when>
					<xsl:otherwise>
						<xsl:value-of select="*[$pos]/@length" />
					</xsl:otherwise>
					</xsl:choose>	
				</xsl:variable>


				<xsl:call-template name="display-child">
					<xsl:with-param name="pos" select="$pos + 1" />
					<xsl:with-param name="offset" select="$offset + ($array-childs-length * $tmpDim1 *
						$tmpDim2 * $tmpDim3 * $tmpDim4 * $testlength)" />
					<xsl:with-param name="representation" select="$representation" />
					<xsl:with-param name="Dim1" select="1" />
					<xsl:with-param name="Dim2" select="1" />
					<xsl:with-param name="Dim3" select="1" />
					<xsl:with-param name="Dim4" select="1" />
					<xsl:with-param name="toplevel" select="$toplevel" /> 
					<xsl:with-param name="subsection" select="$subsection" />
					<xsl:with-param name="bgcolour" select="$bgcolour" />
				</xsl:call-template>				

			</xsl:when>

			<xsl:when test="$class = 'field'">

				<xsl:variable name="length">
					<xsl:call-template name="calculate-length">
						<xsl:with-param name="type" select="*[$pos]/@type" />
						<xsl:with-param name="length" select="*[$pos]/@length" />
						<xsl:with-param name="representation" select="$representation" />
					</xsl:call-template>
				</xsl:variable>
				
				<xsl:variable name="tmpbgcolour">
					<xsl:choose>
					<xsl:when test="*[$pos]/@variable">
						fff0a0
					</xsl:when>
					<xsl:otherwise>
						<xsl:value-of select="$bgcolour" />
					</xsl:otherwise>
					</xsl:choose>
				</xsl:variable>


				<tr valign="top" bgcolor="{concat('#', $tmpbgcolour)}">
					<!-- name -->
					<xsl:variable name="displayname">
						<xsl:choose>
							<xsl:when test="*[$pos]/@name or $name">
								<xsl:value-of select="concat($name, *[$pos]/@name)" />
							</xsl:when>
							<xsl:otherwise>
								spare
							</xsl:otherwise>
						</xsl:choose>
					</xsl:variable>

					<xsl:choose>
						<xsl:when test="*[$pos]/@type = 'bitfield'">						
							<xsl:choose>
								<xsl:when test="$subsection = 1">
									<td align="right">
										<a href="{concat('#', $displayname)}"> <xsl:value-of select="$displayname" /> </a>	
									</td>
								</xsl:when>
								<xsl:otherwise>
									<td>
										<a href="{concat('#', $displayname)}"> <xsl:value-of select="$displayname" /> </a>	
									</td>
								</xsl:otherwise>
							</xsl:choose>
						</xsl:when>
						<xsl:when test="*[$pos]/@type = 'enumerated'">						
							<xsl:choose>
								<xsl:when test="$subsection = 1">
									<td align="right">
										<a href="{concat('#', $displayname)}"> <xsl:value-of select="$displayname" /> </a>	
									</td>
								</xsl:when>
								<xsl:otherwise>
									<td>
										<a href="{concat('#', $displayname)}"> <xsl:value-of select="$displayname" /> </a>	
									</td>
								</xsl:otherwise>
							</xsl:choose>
						</xsl:when>
						<xsl:otherwise>
							<xsl:choose>
								<xsl:when test="$subsection = 1">
									<td align="right">
										<xsl:value-of select="$displayname" />
									</td>
								</xsl:when>
								<xsl:otherwise>
									<td>
										<xsl:value-of select="$displayname" />	
									</td>
								</xsl:otherwise>
							</xsl:choose>							
						</xsl:otherwise>
					</xsl:choose>

					<!-- Description -->
					<td><xsl:value-of select="*[$pos]/@description" /><br/></td>

					<!-- Scaling factor -->
					<td><xsl:value-of select="*[$pos]/@scaling-factor" /><br/></td>

					<!-- Units -->
					<td><xsl:value-of select="*[$pos]/@units" /><br/></td>

					<!-- Dim1 -->
					<td>
					<xsl:choose>
					<xsl:when test="contains($Dim1, '$')">
						<a href="{concat('#', substring($Dim1, 2))}"><xsl:value-of select="substring($Dim1, 2)" /></a>
					</xsl:when>
					<xsl:otherwise>
						<xsl:value-of select="$Dim1" />
					</xsl:otherwise>
					</xsl:choose>					
					</td>

					<!-- Dim2 -->
					<td>
					<xsl:choose>
					<xsl:when test="contains($Dim2, '$')">
						<a href="{concat('#', substring($Dim2, 2))}"><xsl:value-of select="substring($Dim2, 2)" /></a>
					</xsl:when>
					<xsl:otherwise>
						<xsl:value-of select="$Dim2" />
					</xsl:otherwise>
					</xsl:choose>	
					</td>

					<!-- Dim3 -->
					<td>
					<xsl:choose>
					<xsl:when test="contains($Dim3, '$')">
						<a href="{concat('#', substring($Dim3, 2))}"><xsl:value-of select="substring($Dim3, 2)" /></a>
					</xsl:when>
					<xsl:otherwise>
						<xsl:value-of select="$Dim3" />
					</xsl:otherwise>
					</xsl:choose>	
					</td>
		
					<!-- Dim4 -->
					<td>
					<xsl:choose>
					<xsl:when test="contains($Dim4, '$')">
						<a href="{concat('#', substring($Dim4, 2))}"><xsl:value-of select="substring($Dim4, 2)" /></a>
					</xsl:when>
					<xsl:otherwise>
						<xsl:value-of select="$Dim4" />
					</xsl:otherwise>
					</xsl:choose>	
					</td>

					<!-- Type -->
					<td><xsl:value-of select="*[$pos]/@type" />

					<!-- bitfield size in brackets -->
					<xsl:if test="*[$pos]/@type = 'bitfield'">
						(
							<xsl:value-of select="(*[$pos]/@length) div 8" />
						)
					</xsl:if>

					<!-- Integer/uinteger size in brackets (MPHR/SPHR only) -->
					<xsl:if test="$representation = 'text' and *[$pos]/@length">
						<!-- (<xsl:value-of select="*[$pos]/@length" />) -->
					</xsl:if>

					</td>

					
					<!-- Data Type size -->
					<!-- <td><xsl:value-of select="$length" /></td> -->
					 
					<xsl:choose>
						<xsl:when test="$representation = 'text'">
							<td><xsl:value-of select="*[$pos]/@length" /></td>
						</xsl:when>
						<xsl:otherwise>
							<!-- <xsl:if test="$representation = 'binary' and *[$pos]/@length"> -->
								<td><xsl:value-of select="$length" /></td>
							<!-- </xsl:if> -->
						</xsl:otherwise>
					</xsl:choose>

					<!-- Field size -->
					<td><xsl:value-of select="$length * $tmpDim1 * $tmpDim2 * $tmpDim3 * $tmpDim4" /></td>


					<!-- Offset -->
					<xsl:choose>
						<xsl:when test="$subsection = 1">
							<td><font color="White">.</font></td>
						</xsl:when>
						<xsl:otherwise>
							<td><xsl:value-of select="$offset" /></td>
						</xsl:otherwise>
					</xsl:choose>
				</tr>

				<!-- Continue to the next field -->

				<xsl:call-template name="display-child">
					<xsl:with-param name="pos" select="$pos + 1" />
					<xsl:with-param name="offset" select="$offset + $length" />
					<xsl:with-param name="representation" select="$representation" />
					<xsl:with-param name="Dim1" select="1" />
					<xsl:with-param name="Dim2" select="1" />
					<xsl:with-param name="Dim3" select="1" />
					<xsl:with-param name="Dim4" select="1" />
					<xsl:with-param name="toplevel" select="$toplevel" />
					<xsl:with-param name="subsection" select="$subsection" />
					<xsl:with-param name="bgcolour" select="$bgcolour" />
				</xsl:call-template>

			</xsl:when>
		</xsl:choose>
	</xsl:when>

<!--	TOPLEVEL is <xsl:value-of select="$is-top-level" /> <br/> -->


	<xsl:otherwise>
		<xsl:choose>
			<xsl:when test="$toplevel > 3">


				<tr>
					<td colspan="12" align="right">
						Total: <xsl:value-of select="$offset" />
					</td>
				</tr>

			</xsl:when>

<!--
			<xsl:otherwise>
				<tr>
					<td colspan="12" align="right">
						Not printing total because is-top-level is
						<xsl:value-of select="$toplevel" />
					</td>
				</tr>
			</xsl:otherwise>
-->
		</xsl:choose>
	</xsl:otherwise></xsl:choose>


</xsl:template>

<xsl:template name="calculate-length">

	<!-- given an input section return its length in bytes -->

	<xsl:param name="type" />
	<xsl:param name="length" />
	<xsl:param name="representation" />
	<xsl:choose>
		<xsl:when test="$representation = 'text'">
			<xsl:choose>
				<xsl:when test="$type = 'time'"> 
					<xsl:value-of select="15 + 33" />
				</xsl:when>
				<xsl:when test="$type = 'longtime'">
					<xsl:value-of select="18 + 33" />
				</xsl:when>
				<xsl:when test="$type = 'boolean'">
					<xsl:value-of select="1 + 33" />
				</xsl:when>
				<xsl:otherwise>
					<xsl:value-of select="$length + 33"/>
				</xsl:otherwise>
			</xsl:choose>
		</xsl:when>
		<xsl:when test="$representation = 'binary'">
			<xsl:choose>
				<xsl:when test="$type = 'time'"> 
					<xsl:value-of select="6" />
				</xsl:when>
				<xsl:when test="$type = 'longtime'">
					<xsl:value-of select="8" />
				</xsl:when>
				<xsl:when test="$type = 'enumerated'">
					<xsl:value-of select="1" />
				</xsl:when>
				<xsl:when test="$type = 'boolean'">
					<xsl:value-of select="1" />
				</xsl:when>
				<xsl:when test="$type = 'uinteger1'">
					<xsl:value-of select="1" />
				</xsl:when>
				<xsl:when test="$type = 'uinteger2'">
					<xsl:value-of select="2" />
				</xsl:when>
				<xsl:when test="$type = 'uinteger4'">
					<xsl:value-of select="4" />
				</xsl:when>
				<xsl:when test="$type = 'uinteger8'">
					<xsl:value-of select="8" />
				</xsl:when>
				<xsl:when test="$type = 'integer1'">
					<xsl:value-of select="1" />
				</xsl:when>
				<xsl:when test="$type = 'integer2'">
					<xsl:value-of select="2" />
				</xsl:when>
				<xsl:when test="$type = 'integer4'">
					<xsl:value-of select="4" />
				</xsl:when>
				<xsl:when test="$type = 'integer8'">
					<xsl:value-of select="8" />
				</xsl:when>
				<xsl:when test="$type = 'vbyte'">
					<xsl:value-of select="2" />
				</xsl:when>
				<xsl:when test="$type = 'vinteger2'">
					<xsl:value-of select="3" />
				</xsl:when>
				<xsl:when test="$type = 'vinteger4'">
					<xsl:value-of select="5" />
				</xsl:when>
				<xsl:when test="$type = 'vinteger8'">
					<xsl:value-of select="9" />
				</xsl:when>
				<xsl:when test="$type = 'vubyte'">
					<xsl:value-of select="2" />
				</xsl:when>
				<xsl:when test="$type = 'vuinteger2'">
					<xsl:value-of select="3" />
				</xsl:when>
				<xsl:when test="$type = 'vuinteger4'">
					<xsl:value-of select="5" />
				</xsl:when>
				<xsl:when test="$type = 'vuinteger8'">
					<xsl:value-of select="9" />
				</xsl:when>
				<xsl:when test="$type = 'string'">
					<xsl:value-of select="$length" />
				</xsl:when>
				<xsl:when test="$type = 'bitfield'">
					<xsl:value-of select="$length div 8" />
				</xsl:when>
			</xsl:choose>
		</xsl:when>

		<xsl:otherwise>
			ERROR - calculate_length called with no representation
		</xsl:otherwise>

	</xsl:choose>	
</xsl:template>

<xsl:template name="recursive-calculate-length">

	<xsl:param name="representation" />
	<xsl:param name="pos" select="1" />

<!--	<xsl:if test="count(*) >= $pos"> -->
<!--
	NOM <xsl:value-of select="@name" /> MON
	POS <xsl:value-of select="$pos" /> SOP

	a <xsl:value-of select="*[1]/@name"/> a
	b <xsl:value-of select="*[2]/@name"/> b
-->

	<xsl:variable name="class">
		<xsl:for-each select="*[$pos]">
			<xsl:value-of select="name()" />
		</xsl:for-each>
	</xsl:variable>

<!--	CLASS <xsl:value-of select="$class" /> SSALC -->
	
	<xsl:choose>

		<xsl:when test="$class = 'field'">

			<xsl:variable name="this-item">
				<xsl:call-template name="calculate-length" >
					<xsl:with-param name="type" select="*[$pos]/@type" />
					<xsl:with-param name="length" select="*[$pos]/@length" />
					<xsl:with-param name="representation" select="$representation" />
				</xsl:call-template>
			</xsl:variable>
			<xsl:variable name="next-item">
				<xsl:choose>
					<xsl:when test="count(*) > $pos">
						<xsl:call-template name="recursive-calculate-length">
							<xsl:with-param name="representation" select="$representation" />
							<xsl:with-param name="pos" select="$pos + 1" />
						</xsl:call-template>
					</xsl:when>
					<xsl:otherwise>
						<xsl:value-of select="0" />
					</xsl:otherwise>
				</xsl:choose>
			</xsl:variable>
			<xsl:value-of select="$this-item + $next-item" />


<!--	<xsl:value-of select="100" /> -->

		</xsl:when>

		<xsl:when test="$class = 'section'">

			<xsl:variable name="childs">
				<xsl:for-each select="*[$pos]">
					<xsl:call-template name="recursive-calculate-length">
						<xsl:with-param name="representation" select="$representation" />
					</xsl:call-template>
				</xsl:for-each>
			</xsl:variable>
			<xsl:variable name="next-item">
				<xsl:call-template name="recursive-calculate-length">
					<xsl:with-param name="representation" select="$representation" />
					<xsl:with-param name="pos" select="$pos + 1" />
				</xsl:call-template>
			</xsl:variable>
			<xsl:value-of select="$childs + $next-item" />

		</xsl:when>

		<xsl:when test="$class = 'array'">

			<xsl:variable name="child">
				<xsl:for-each select="*[$pos]">
					<xsl:call-template name="recursive-calculate-length">
						<xsl:with-param name="representation" select="$representation" />
					</xsl:call-template>
				</xsl:for-each>
			</xsl:variable>
			<xsl:variable name="next-item">
				<xsl:call-template name="recursive-calculate-length">
					<xsl:with-param name="representation" select="$representation" />
					<xsl:with-param name="pos" select="$pos + 1" />
				</xsl:call-template>
			</xsl:variable>

			<!-- length -->
			<xsl:variable name="tmplength">
				<xsl:choose>
				<xsl:when test="contains(*[$pos]/@length, '$')">
					<xsl:call-template name="getparamvalue">
						<xsl:with-param name="name" select="*[$pos]/@length" />
					</xsl:call-template>
				</xsl:when>
				<xsl:otherwise>
					<xsl:value-of select="*[$pos]/@length" />
				</xsl:otherwise>
				</xsl:choose>	
			</xsl:variable>

			<xsl:value-of select="($child * $tmplength) + $next-item" />

		</xsl:when>
			

		<xsl:otherwise>
			<xsl:value-of select="0"/>

		</xsl:otherwise>
	
	</xsl:choose>

</xsl:template>


<xsl:template name="doenumeratedtest">
	<xsl:param name="pos" />
	<xsl:param name="name" />
	<xsl:param name="type" />

	<!--
		<br/>
		<xsl:text>type = </xsl:text>
		<xsl:value-of select="string($type)"/>
		<br/>
		<xsl:text>pos = </xsl:text>
		<xsl:value-of select="$pos"/>
		<br/>
		<xsl:text>name = </xsl:text>
		<xsl:value-of select="$name"/>
	-->


	<xsl:choose>
		<xsl:when test="string($type) = string('field')">
			<xsl:choose>
				<xsl:when test="$pos = 1">
					<xsl:choose>
						<xsl:when test="string($name) = string('')">
							<!-- Do nothing -->
						</xsl:when>
						<xsl:otherwise>
							<xsl:call-template name="doenumerated">
								<xsl:with-param name="pos" select="$pos" />
								<xsl:with-param name="name" select="$name" />
								<xsl:with-param name="type" select="$type" />
							</xsl:call-template>
						</xsl:otherwise>
					</xsl:choose>
				</xsl:when>
				<xsl:otherwise>

					<xsl:for-each select="//field[@type='enumerated']">
						<xsl:sort select="@name" order="ascending"/>
						<xsl:if test="position() = $pos - 1">
						<xsl:choose>
							<xsl:when test="string(@name) = string($name)">
								<!-- Do nothing -->
							</xsl:when>
							<xsl:otherwise>
								<xsl:call-template name="doenumerated">
									<xsl:with-param name="pos" select="$pos" />
									<xsl:with-param name="name" select="$name" />
									<xsl:with-param name="type" select="$type" />
								</xsl:call-template>
							</xsl:otherwise>
						</xsl:choose>
						</xsl:if>
					</xsl:for-each>
				
				</xsl:otherwise>
			</xsl:choose>
		</xsl:when>
		<xsl:when test="string($type) = string('array')">
			<xsl:for-each select="//array">
				<xsl:sort select="@name" order="ascending"/>
				<xsl:variable name="tmpname" select="@name"/>
				<xsl:variable name="tmppos" select="position()"/>
				<xsl:for-each select="field[@type='enumerated']">
					<xsl:if test="$tmppos = $pos">
						<xsl:variable name="prename">
							<xsl:call-template name="previousenumerated">
								<xsl:with-param name="pos" select="$tmppos" />
							</xsl:call-template>
						</xsl:variable>
						
						<!--
						<br/><xsl:text>$previousname=</xsl:text><xsl:value-of select="$prename" />
						-->

						<xsl:choose>
							<xsl:when test="string($tmpname) = string($prename)">
								<!-- Do Nothing -->
							</xsl:when>
							<xsl:otherwise>
								<xsl:call-template name="doenumerated">
									<xsl:with-param name="pos" select="$pos" />
									<xsl:with-param name="name" select="$name" />
									<xsl:with-param name="type" select="$type" />
								</xsl:call-template>
							</xsl:otherwise>
						</xsl:choose>
					</xsl:if>
				</xsl:for-each>
			</xsl:for-each>
		</xsl:when>
		<xsl:when test="string($type) = string('array2')">
			<xsl:for-each select="//array">
				<xsl:sort select="@name" order="ascending"/>
				<xsl:variable name="tmpname" select="@name"/>
				<xsl:variable name="tmppos" select="position()"/>
				<xsl:for-each select="array">
				<xsl:for-each select="field[@type='enumerated']">
					<xsl:if test="$tmppos = $pos">
						<xsl:variable name="prename">
							<xsl:call-template name="previousenumerated">
								<xsl:with-param name="pos" select="$tmppos" />
								<xsl:with-param name="type" select="$type" />
							</xsl:call-template>
						</xsl:variable>
						
						<!--
						<br/><xsl:text>$previousname=</xsl:text><xsl:value-of select="$prename" />
						-->

						<xsl:choose>
							<xsl:when test="string($tmpname) = string($prename)">
								<!-- Do Nothing -->
							</xsl:when>
							<xsl:otherwise>
								<xsl:call-template name="doenumerated">
									<xsl:with-param name="pos" select="$pos" />
									<xsl:with-param name="name" select="$name" />
									<xsl:with-param name="type" select="$type" />
								</xsl:call-template>
							</xsl:otherwise>
						</xsl:choose>
					</xsl:if>
				</xsl:for-each>
				</xsl:for-each>
			</xsl:for-each>
		</xsl:when>
	</xsl:choose>

</xsl:template>


<xsl:template name="doenumerated">
	<xsl:param name="pos" />
	<xsl:param name="name" />
	<xsl:param name="type" />

	<!-- add an Enumerated section to the appendix -->


	<xsl:choose>
		<xsl:when test="string($type) = string('field')">
			<xsl:for-each select="//field[@type='enumerated']">
				<xsl:sort select="@name" order="ascending"/>
				<xsl:if test="position() = $pos">
						<h3>
							Enumeration
							<a name="{concat(@name,'')}" > <xsl:value-of select="@name" /> </a>
						</h3>
						<table border="1">
							<tr bgcolor="#a0a0a0">
								<td>Value</td>
								<td>Name</td>
								<td>Description</td>				
							</tr>

						<xsl:for-each select="item">
							<tr valign="top">
							<td>
							<xsl:choose>
							<xsl:when test="@value">
								<xsl:value-of select="@value" />
							</xsl:when>
							<xsl:otherwise>
								<font color="White">.</font>
							</xsl:otherwise>
							</xsl:choose>
							</td>

							<!-- <td><xsl:value-of select="@name" /></td> -->
							<td>
							<xsl:choose>
							<xsl:when test="@name">
								<xsl:value-of select="@name" />
							</xsl:when>
							<xsl:otherwise>
								<font color="White">.</font>
							</xsl:otherwise>
							</xsl:choose>
							</td>

							<td>
							<xsl:choose>
							<xsl:when test="@description">
								<xsl:value-of select="@description" />
							</xsl:when>
							<xsl:otherwise>
								<font color="White">.</font>
							</xsl:otherwise>
							</xsl:choose>
							</td>
			
							</tr>
						</xsl:for-each>
						</table>
				</xsl:if>
			</xsl:for-each>
		</xsl:when>
		<xsl:when test="string($type) = string('array')">
			<xsl:for-each select="//array">
				<xsl:sort select="@name" order="ascending"/>
				<xsl:variable name="tmpname">
					<xsl:value-of select="@name" />
				</xsl:variable>
				<xsl:variable name="tmppos">
					<xsl:value-of select="position()" />
				</xsl:variable>
				<xsl:for-each select="field[@type='enumerated']">
					<xsl:if test="$tmppos = $pos">
						<h3>
							Enumeration
							<a name="{concat($tmpname,'')}" > <xsl:value-of select="$tmpname" /> </a>
						</h3>
						<table border="1">
							<tr bgcolor="#a0a0a0">
								<td>Value</td>
								<td>Name</td>
								<td>Description</td>				
							</tr>

						<xsl:for-each select="item">
							<tr valign="top">
							<td>
							<xsl:choose>
							<xsl:when test="@value">
								<xsl:value-of select="@value" />
							</xsl:when>
							<xsl:otherwise>
								<font color="White">.</font>
							</xsl:otherwise>
							</xsl:choose>
							</td>

							<!-- <td><xsl:value-of select="@name" /></td> -->
							<td>
							<xsl:choose>
							<xsl:when test="@name">
								<xsl:value-of select="@name" />
							</xsl:when>
							<xsl:otherwise>
								<font color="White">.</font>
							</xsl:otherwise>
							</xsl:choose>
							</td>
			
							<td>
							<xsl:choose>
							<xsl:when test="@description">
								<xsl:value-of select="@description" />
							</xsl:when>
							<xsl:otherwise>
								<font color="White">.</font>
							</xsl:otherwise>
							</xsl:choose>
							</td>
			
							</tr>
						</xsl:for-each>
						</table>
					</xsl:if>
				</xsl:for-each>
			</xsl:for-each>
		</xsl:when>
		<xsl:when test="string($type) = string('array2')">
			<xsl:for-each select="//array">
				<xsl:sort select="@name" order="ascending"/>
				<xsl:variable name="tmpname">
					<xsl:value-of select="@name" />
				</xsl:variable>
				<xsl:variable name="tmppos">
					<xsl:value-of select="position()" />
				</xsl:variable>
				<xsl:for-each select="array">
				<xsl:for-each select="field[@type='enumerated']">
					<xsl:if test="$tmppos = $pos">
						<h3>
							Enumeration
							<a name="{concat($tmpname,'')}" > <xsl:value-of select="$tmpname" /> </a>
						</h3>
						<table border="1">
							<tr bgcolor="#a0a0a0">
								<td>Value</td>
								<td>Name</td>
								<td>Description</td>				
							</tr>

						<xsl:for-each select="item">
							<tr valign="top">
							<td>
							<xsl:choose>
							<xsl:when test="@value">
								<xsl:value-of select="@value" />
							</xsl:when>
							<xsl:otherwise>
								<font color="White">.</font>
							</xsl:otherwise>
							</xsl:choose>
							</td>

							<!-- <td><xsl:value-of select="@name" /></td> -->
							<td>
							<xsl:choose>
							<xsl:when test="@name">
								<xsl:value-of select="@name" />
							</xsl:when>
							<xsl:otherwise>
								<font color="White">.</font>
							</xsl:otherwise>
							</xsl:choose>
							</td>
			
							<td>
							<xsl:choose>
							<xsl:when test="@description">
								<xsl:value-of select="@description" />
							</xsl:when>
							<xsl:otherwise>
								<font color="White">.</font>
							</xsl:otherwise>
							</xsl:choose>
							</td>
			
							</tr>
						</xsl:for-each>
						</table>
					</xsl:if>
				</xsl:for-each>
				</xsl:for-each>
			</xsl:for-each>
		</xsl:when>
	</xsl:choose>

</xsl:template>


<xsl:template name="dobitfieldtest">
	<xsl:param name="bitfieldpos" />
	<xsl:param name="bitfieldname" />
	<xsl:param name="bitfieldtype" />

	<!--
		<br/>
		<xsl:text>bitfieldtype = </xsl:text>
		<xsl:value-of select="string($bitfieldtype)"/>
		<br/>
		<xsl:text>bitfieldpos = </xsl:text>
		<xsl:value-of select="$bitfieldpos"/>
		<br/>
		<xsl:text>bitfieldname = </xsl:text>
		<xsl:value-of select="$bitfieldname"/>
	-->


	<xsl:choose>
		<xsl:when test="string($bitfieldtype) = string('field')">
			<xsl:choose>
				<xsl:when test="$bitfieldpos = 1">
					<xsl:choose>
						<xsl:when test="string($bitfieldname) = string('')">
							<!-- Do nothing -->
						</xsl:when>
						<xsl:otherwise>
							<xsl:call-template name="dobitfield">
								<xsl:with-param name="bitfieldpos" select="$bitfieldpos" />
								<xsl:with-param name="bitfieldname" select="$bitfieldname" />
								<xsl:with-param name="bitfieldtype" select="$bitfieldtype" />
							</xsl:call-template>
						</xsl:otherwise>
					</xsl:choose>
				</xsl:when>
				<xsl:otherwise>

					<xsl:for-each select="//field[@type='bitfield']">
						<xsl:sort select="@name" order="ascending"/>
				
						<!-- 
							<br/>
							<xsl:text>bitfieldpos2 = </xsl:text>
							<xsl:value-of select="position()"/>
							<br/>
							<xsl:text>bitfieldname2 = </xsl:text>
							<xsl:value-of select="@name"/>
						-->

						<xsl:if test="position() = $bitfieldpos - 1">
						<xsl:choose>
							<xsl:when test="string(@name) = string($bitfieldname)">
								<!-- Do nothing -->
							</xsl:when>
							<xsl:otherwise>
								<!--
									<xsl:text>@name = </xsl:text>
									<xsl:value-of select="@name"/>
									<xsl:text>$bitfieldname = </xsl:text>
									<xsl:value-of select="$bitfieldname"/>
								-->
								<xsl:call-template name="dobitfield">
									<xsl:with-param name="bitfieldpos" select="$bitfieldpos" />
									<xsl:with-param name="bitfieldname" select="$bitfieldname" />
									<xsl:with-param name="bitfieldtype" select="$bitfieldtype" />
								</xsl:call-template>
							</xsl:otherwise>
						</xsl:choose>
						</xsl:if>
					</xsl:for-each>
				
				</xsl:otherwise>
			</xsl:choose>
		</xsl:when>
		<xsl:when test="string($bitfieldtype) = string('array')">

			<!--
			<br/>
			<xsl:text>name=</xsl:text><xsl:value-of select="$bitfieldname"/>
			<br/>
			<xsl:text>pos=</xsl:text><xsl:value-of select="$bitfieldpos"/>
			-->

			<xsl:for-each select="//array">
				<xsl:sort select="@name" order="ascending"/>
				<xsl:variable name="tmpname" select="@name"/>
				<xsl:variable name="namepos" select="position()"/>
				<xsl:for-each select="field[@type='bitfield']">
					<xsl:if test="$namepos = $bitfieldpos">
						<xsl:variable name="prename">
							<xsl:call-template name="previousbitfield">
								<xsl:with-param name="pos" select="$namepos" />
								<xsl:with-param name="type" select="$bitfieldtype" />
							</xsl:call-template>
						</xsl:variable>
						
						<!--
						<br/><xsl:text>$previousname=</xsl:text><xsl:value-of select="$prename" />
						-->

						<xsl:choose>
							<xsl:when test="string($tmpname) = string($prename)">
								<!-- Do Nothing -->
							</xsl:when>
							<xsl:otherwise>
								<xsl:call-template name="dobitfield">
									<xsl:with-param name="bitfieldpos" select="$bitfieldpos" />
									<xsl:with-param name="bitfieldname" select="$bitfieldname" />
									<xsl:with-param name="bitfieldtype" select="$bitfieldtype" />
								</xsl:call-template>
							</xsl:otherwise>
						</xsl:choose>
					</xsl:if>
				</xsl:for-each>
			</xsl:for-each>
		</xsl:when>
		<xsl:when test="string($bitfieldtype) = string('array2')">
			<xsl:for-each select="//array">
				<xsl:sort select="@name" order="ascending"/>
				<xsl:variable name="tmpname" select="@name"/>
				<xsl:variable name="namepos" select="position()"/>
				<xsl:for-each select="array">
				<xsl:for-each select="field[@type='bitfield']">
					<xsl:if test="$namepos = $bitfieldpos">
						<xsl:variable name="prename">
							<xsl:call-template name="previousbitfield">
								<xsl:with-param name="pos" select="$namepos" />
								<xsl:with-param name="type" select="$bitfieldtype" />
							</xsl:call-template>
						</xsl:variable>
						
						<!--
						<br/><xsl:text>$previousname=</xsl:text><xsl:value-of select="$prename" />
						-->

						<xsl:choose>
							<xsl:when test="string($tmpname) = string($prename)">
								<!-- Do Nothing -->
							</xsl:when>
							<xsl:otherwise>
								<xsl:call-template name="dobitfield">
									<xsl:with-param name="bitfieldpos" select="$bitfieldpos" />
									<xsl:with-param name="bitfieldname" select="$bitfieldname" />
									<xsl:with-param name="bitfieldtype" select="$bitfieldtype" />
								</xsl:call-template>
							</xsl:otherwise>
						</xsl:choose>
					</xsl:if>
				</xsl:for-each>
				</xsl:for-each>
			</xsl:for-each>
		</xsl:when>
	</xsl:choose>

</xsl:template>


<xsl:template name="dobitfield">
	<xsl:param name="bitfieldpos" />
	<xsl:param name="bitfieldname" />
	<xsl:param name="bitfieldtype"/>

	<!-- add a Bitfield section to the appendix -->
	
	<!--
	<xsl:value-of select="$bitfieldpos"/>	
	<xsl:value-of select="$bitfieldname"/>
	<xsl:value-of select="$bitfieldtype"/>
	-->

	<xsl:choose>
		<xsl:when test="string($bitfieldtype) = string('field')">
			<xsl:for-each select="//field[@type='bitfield']">
				<xsl:sort select="@name" order="ascending"/>
				<xsl:if test="position() = $bitfieldpos">

					<h3>
						Bitfield
						<a name="{concat(@name,'')}" > <xsl:value-of select="@name" /> </a>
					</h3>
					<i>
						Length 
							<xsl:value-of select="@length div 8" />
						bytes
					</i><br/><br/>
					<table border="1">
						<tr bgcolor="#a0a0a0">
							<td>Name</td>
							<td>Description</td>
							<td>Length</td>
						</tr>

						<xsl:call-template name="printbitfield">
							<xsl:with-param name="pos" select="1" />
						</xsl:call-template>

						<tr>
							<td>Total</td>
							<td><font color="White">.</font></td>
							<td><xsl:value-of select="@length" /></td>
						</tr>
					</table>

				</xsl:if>
			</xsl:for-each>
		</xsl:when>
		<xsl:when test="string($bitfieldtype) = string('array')">
			<xsl:for-each select="//array">
				<xsl:sort select="@name" order="ascending"/>
				<xsl:variable name="tmpname">
					<xsl:value-of select="@name" />
				</xsl:variable>
				<xsl:variable name="tmppos">
					<xsl:value-of select="position()" />
				</xsl:variable>
				<xsl:for-each select="field[@type='bitfield']">
					<xsl:if test="$tmppos = $bitfieldpos">

						<h3>
							Bitfield
							<a name="{concat($bitfieldname,'')}" > <xsl:value-of select="$bitfieldname" /> </a>
						</h3>
						<i>
							Length 
								<xsl:value-of select="@length div 8" />
							bytes
						</i><br/><br/>
						<table border="1">
							<tr bgcolor="#a0a0a0">
								<td>Name</td>
								<td>Description</td>
								<td>Length</td>
							</tr>

							<xsl:call-template name="printbitfield">
								<xsl:with-param name="pos" select="1" />
							</xsl:call-template>
	
							<tr>
								<td>Total</td>
								<td><font color="White">.</font></td>
								<td><xsl:value-of select="@length" /></td>
							</tr>
						</table>

					</xsl:if>
				</xsl:for-each>
			</xsl:for-each>
		</xsl:when>
		<xsl:when test="string($bitfieldtype) = string('array2')">
			<xsl:for-each select="//array">
				<xsl:sort select="@name" order="ascending"/>
				<xsl:variable name="tmpname">
					<xsl:value-of select="@name" />
				</xsl:variable>
				<xsl:variable name="tmppos">
					<xsl:value-of select="position()" />
				</xsl:variable>
				<xsl:for-each select="array">
				<xsl:for-each select="field[@type='bitfield']">
					<xsl:if test="$tmppos = $bitfieldpos">

						<h3>
							Bitfield
							<a name="{concat($bitfieldname,'')}" > <xsl:value-of select="$bitfieldname" /> </a>
						</h3>
						<i>
							Length 
								<xsl:value-of select="@length div 8" />
							bytes
						</i><br/><br/>
						<table border="1">
							<tr bgcolor="#a0a0a0">
								<td>Name</td>
								<td>Description</td>
								<td>Length</td>
							</tr>

							<xsl:call-template name="printbitfield">
								<xsl:with-param name="pos" select="1" />
							</xsl:call-template>
	
							<tr>
								<td>Total</td>
								<td><font color="White">.</font></td>
								<td><xsl:value-of select="@length" /></td>
							</tr>
						</table>

					</xsl:if>
				</xsl:for-each>
				</xsl:for-each>
			</xsl:for-each>
		</xsl:when>
	</xsl:choose>

</xsl:template>


<xsl:template name="printbitfield">
	<xsl:param name="pos" />

	<xsl:for-each select="*">

	<!--
		<br/>
		<xsl:text>pos1 = </xsl:text>
		<xsl:value-of select="position()"/>
		<br/>
		<xsl:text>pso2 = </xsl:text>
		<xsl:value-of select="$pos"/>
	-->

		<xsl:variable name="test">
			<xsl:value-of select="name()" />
		</xsl:variable>

		<xsl:choose>

			<xsl:when test="$pos = 2">
				<!-- Do nothing -->
			</xsl:when>

			<xsl:when test="$test = 'array'">
				<tr>
				<td>
				<xsl:choose>
					<xsl:when test="@name">
						<xsl:value-of select="@name" />
					</xsl:when>
					<xsl:otherwise>
						<font color="White">.</font>
					</xsl:otherwise>
				</xsl:choose>
				</td>
				<xsl:for-each select="bit">
					<td>
					<xsl:choose>
						<xsl:when test="@description">
							<xsl:value-of select="@description" />
						</xsl:when>
						<xsl:otherwise>
							<font color="White">.</font>
						</xsl:otherwise>
					</xsl:choose>
					</td>
				</xsl:for-each>
				<xsl:for-each select="bits">
					<td>
					<xsl:choose>
						<xsl:when test="@description">
							<xsl:value-of select="@description" />
						</xsl:when>
						<xsl:otherwise>
							<font color="White">.</font>
						</xsl:otherwise>
					</xsl:choose>
					</td>
				</xsl:for-each>
				<td>
				<xsl:choose>
					<xsl:when test="@length">
						<xsl:value-of select="@length" />
					</xsl:when>
					<xsl:otherwise>
						1
					</xsl:otherwise>
				</xsl:choose>
				</td>
				</tr>
				<xsl:call-template name="printbitfield">
					<xsl:with-param name="pos" select="$pos + 1" />
				</xsl:call-template>
			</xsl:when>

			<xsl:when test="$test = 'bit'">
				<tr>
				<td>
				<xsl:choose>
					<xsl:when test="@name">
						<xsl:value-of select="@name" />
					</xsl:when>
					<xsl:otherwise>
						<font color="White">.</font>
					</xsl:otherwise>
				</xsl:choose>
				</td>
				<td>
				<xsl:choose>
					<xsl:when test="@description">
						<xsl:value-of select="@description" />
					</xsl:when>
					<xsl:otherwise>
						<font color="White">.</font>
					</xsl:otherwise>
					</xsl:choose>
					</td>
					<td>
				<xsl:choose>
					<xsl:when test="@length">
						<xsl:value-of select="@length" />
					</xsl:when>
					<xsl:otherwise>
						1
					</xsl:otherwise>
				</xsl:choose>
				</td>
				</tr>
				<xsl:call-template name="printbitfield">
					<xsl:with-param name="pos" select="$pos + 1" />
				</xsl:call-template>
			</xsl:when>

			<xsl:when test="$test = 'bits'">
				<tr>
				<td>
				<xsl:choose>
					<xsl:when test="@name">
						<xsl:value-of select="@name" />
					</xsl:when>
					<xsl:otherwise>
						<font color="White">.</font>
					</xsl:otherwise>
				</xsl:choose>
				</td>
				<td>
				<xsl:choose>
					<xsl:when test="@description">
						<xsl:value-of select="@description" />
					</xsl:when>
					<xsl:otherwise>
						<font color="White">.</font>
					</xsl:otherwise>
					</xsl:choose>
					</td>
					<td>
				<xsl:choose>
					<xsl:when test="@length">
						<xsl:value-of select="@length" />
					</xsl:when>
					<xsl:otherwise>
						1
					</xsl:otherwise>
				</xsl:choose>
				</td>
				</tr>
				<xsl:call-template name="printbitfield">
					<xsl:with-param name="pos" select="$pos + 1" />
				</xsl:call-template>
			</xsl:when>

			<xsl:otherwise>
				<!-- End call back loop -->
			</xsl:otherwise>

		</xsl:choose>

	</xsl:for-each>

</xsl:template>


<xsl:template name="display-toplevel-intro">
	<xsl:param name="classname" />
	<xsl:param name="classnumber" />
	<xsl:param name="subclassnumber" />
	<xsl:param name="name" ></xsl:param>
	<xsl:param name="version"/>

	<xsl:variable name="linkName">
		<xsl:value-of select="$classname" />
		(
		<xsl:if test="$name">
			name '<xsl:value-of select="$name" />',
		</xsl:if>
		<xsl:if test="$subclassnumber">
			subclass <xsl:value-of select="$subclassnumber" />,
		</xsl:if>
		version <xsl:value-of select="$version" />
		)
	</xsl:variable>

	<h3>
		<a name="{concat($name,'')}" > <xsl:value-of select="$linkName" /> </a>		
	</h3>
	<i>
		class <xsl:value-of select="$classnumber" />
	</i>
	<br/>
	<br/>

</xsl:template>

<xsl:template name="display-table-banner">

	<thead>
	<tr bgcolor="#a0a0a0" valign="top">
		<td>Name</td>
		<td>Description</td>
		<td>Scaling factor</td>
		<td>Units</td>
		<td>Dim1</td>
		<td>Dim2</td>
		<td>Dim3</td>
		<td>Dim4</td>
		<td>Type</td>
		<td>Type size</td>
		<td>Field size</td>
		<td>Offset</td>
	</tr>
	</thead>
	
	<tbody>
	<tr valign="top">
		<td>RECORD_HEADER</td>
		<td>Generic Record Header</td>
		<td><font color="White">.</font></td>
		<td><font color="White">.</font></td>
		<td>1</td>
		<td>1</td>
		<td>1</td>
		<td>1</td>
		<td>REC_HEAD</td>
		<td>20</td>
		<td>20</td>
		<td>0</td>
	</tr>
	</tbody>

</xsl:template>

<xsl:template name="display-eadr">
	<xsl:param name="classname" />
	<xsl:param name="classnumber" />
	<xsl:param name="subclassnumber" />
	<xsl:param name="name" />
	<xsl:param name="version" />


	<xsl:call-template name="display-toplevel-intro">
		<xsl:with-param name="classname" select="$classname" />
		<xsl:with-param name="classnumber" select="$classnumber" />
		<xsl:with-param name="subclassnumber" select="$subclassnumber" />
		<xsl:with-param name="name" select="$name" />
		<xsl:with-param name="version" select="$version" />
	</xsl:call-template>


	<table border="1">

		<xsl:call-template name="display-table-banner" />	

		<tr>
			<td>Identifier</td>
			<td>Unique pointer to auxiliary dataset</td>
			<td><font color="White">.</font></td>
			<td><font color="White">.</font></td>
			<td>1</td>
			<td>1</td>
			<td>1</td>
			<td>1</td>
			<td>POINTER</td>
			<td>100</td>
			<td>100</td>
			<td>120</td>
		</tr>
	
		<tr>
			<td colspan="12" align="right">
				Total: 120
			</td>
		</tr>

	</table>

</xsl:template>			


<xsl:template name="previousbitfield">
	<xsl:param name="pos" />
	<xsl:param name="type" />
	
	<xsl:choose>
	<xsl:when test="$type = 'array'">
		<xsl:for-each select="//array">
		<xsl:sort select="@name" order="ascending"/>
		<xsl:variable name="tmpname" select="@name"/>
		<xsl:variable name="tmppos" select="position()"/>
		<xsl:for-each select="field[@type='bitfield']">
			<xsl:if test="$tmppos = $pos - 1">
				<xsl:value-of select="$tmpname" />
			</xsl:if>
		</xsl:for-each>
		</xsl:for-each>
	</xsl:when>
	<xsl:when test="$type = 'array2'">
		<xsl:for-each select="//array">
		<xsl:sort select="@name" order="ascending"/>
		<xsl:variable name="tmpname" select="@name"/>
		<xsl:variable name="tmppos" select="position()"/>
		<xsl:for-each select="array">
		<xsl:for-each select="field[@type='bitfield']">
				<xsl:if test="$tmppos = $pos - 1">
			<xsl:value-of select="$tmpname" />
			</xsl:if>
		</xsl:for-each>
		</xsl:for-each>
		</xsl:for-each>
	</xsl:when>
	</xsl:choose>

</xsl:template>


<xsl:template name="previousenumerated">
	<xsl:param name="pos" />
	<xsl:param name="type" />
	
	<xsl:choose>
	<xsl:when test="$type = 'array'">
		<xsl:for-each select="//array">
		<xsl:sort select="@name" order="ascending"/>
		<xsl:variable name="tmpname" select="@name"/>
		<xsl:variable name="tmppos" select="position()"/>
		<xsl:for-each select="field[@type='enumerated']">
				<xsl:if test="$tmppos = $pos - 1">
			<xsl:value-of select="$tmpname" />
			</xsl:if>
		</xsl:for-each>
		</xsl:for-each>
	</xsl:when>
	<xsl:when test="$type = 'array2'">
		<xsl:for-each select="//array">
		<xsl:sort select="@name" order="ascending"/>
		<xsl:variable name="tmpname" select="@name"/>
		<xsl:variable name="tmppos" select="position()"/>
		<xsl:for-each select="array">
		<xsl:for-each select="field[@type='enumerated']">
				<xsl:if test="$tmppos = $pos - 1">
			<xsl:value-of select="$tmpname" />
			</xsl:if>
		</xsl:for-each>
		</xsl:for-each>
		</xsl:for-each>
	</xsl:when>
	</xsl:choose>

</xsl:template>

<xsl:template name="getparamvalue">
	<xsl:param name="name" />

	<xsl:for-each select="//parameters">
		<xsl:for-each select="parameter">
			<xsl:if test="$name = concat('$', @name)">
				<xsl:value-of select="@value" />
			</xsl:if>		
		</xsl:for-each>
	</xsl:for-each>

</xsl:template>


<xsl:template name="display-parameters">

	<h3>
		Parameters Table
	</h3>
	<table border="1">
		<tr bgcolor="#a0a0a0">
			<td>Parameter</td>
			<td>Value</td>
			<td>Description</td>				
		</tr>

		<xsl:for-each select="parameter">
			<tr valign="top">

			<td>
			<xsl:choose>
				<xsl:when test="@name">
					<a name="{concat(@name, '')}" > <xsl:value-of select="@name" /> </a>
				</xsl:when>
				<xsl:otherwise>
					<br/>
				</xsl:otherwise>
			</xsl:choose>
			</td>

			<td>
			<xsl:choose>
				<xsl:when test="@value">
					<xsl:value-of select="@value" />
				</xsl:when>
				<xsl:otherwise>
					<br/>
				</xsl:otherwise>
			</xsl:choose>
			</td>
			
			<td>
			<xsl:choose>
				<xsl:when test="@description">
					<xsl:value-of select="@description" />
				</xsl:when>
				<xsl:otherwise>
					<br/>
				</xsl:otherwise>
			</xsl:choose>
			</td>
			
			</tr>
		</xsl:for-each>
	</table>

</xsl:template>


<xsl:template name="display-contents">
	<xsl:param name="classname" />
	<xsl:param name="classnumber" />
	<xsl:param name="subclassnumber" />
	<xsl:param name="name"></xsl:param>
	<xsl:param name="version"/>

	<xsl:variable name="linkName">
		<xsl:value-of select="$classname" />
		(
		<xsl:if test="$name">
			name '<xsl:value-of select="$name" />',
		</xsl:if>
		<xsl:if test="$subclassnumber">
			subclass <xsl:value-of select="$subclassnumber" />,
		</xsl:if>
		version <xsl:value-of select="$version" />
		)
	</xsl:variable>

	<li>
	<a href="{concat('#', $name)}"><xsl:value-of select="$linkName" /></a>
	</li>

</xsl:template>



</xsl:stylesheet>


